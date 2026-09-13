#!/usr/bin/env python3
"""Turn a colour image into a printable cross-stitch symbol chart.

Takes any image, resamples it to a stitch grid, matches every stitch to the
nearest DMC stranded-cotton colour, assigns a symbol per colour, and writes:

  * tiled chart pages, each a grid of big squares, named by x/y position
  * a colour key (symbol -> DMC code -> name -> swatch)
  * a CSV of the raw symbol matrix, so the decode is auditable
  * an optional colour preview of what the finished piece should look like

Two modes, depending on whether a symbol key already exists:

  Kit chart (--palette KEY.csv)
      You already own the pattern and its printed key. Pass the key and the
      script reuses its exact symbol -> DMC mapping, so the notation matches
      the paper chart. Threads keep their symbol even when unused, because the
      notation has to stay stable against the printed key.

  Original design (no --palette)
      No key exists yet, so the script picks the palette itself, capped by
      --colors, and letters it by area: the commonest colour gets the clearest
      glyph. The mapping is specific to that one run.

Either way the grid geometry, tiling and page numbering are identical -- only
where the symbols come from differs.

Colour matching runs in CIELAB with a perceptual distance, not raw RGB --
RGB-Euclidean matching picks visibly wrong threads, especially in skin tones
and muted greens.

Tile naming: the origin is the TOP-LEFT of the charted region. x increases to
the right, y increases downward, both 1-based. So `chart-x1y1` is the top-left
page, `chart-x2y1` sits immediately to its right, and `chart-x1y2` sits
immediately below it.

Page size is set by --squares-per-page, counted in big squares (each --block
stitches). Pass one number for a square page or WxH for different across/down
counts: `3` gives 3x3 squares (30x30 stitches), `6x8` gives 60x80.

Each page also carries context for the seams: the first two stitches of each
adjoining page, drawn as real symbol cells but greyed and set off by a gap, so
notation can be matched across a seam without unfolding the next sheet. The
neighbouring page's name is printed alongside its band, and a small locator map
of the whole chart marks the current page. Turn it off with --no-context.

Usage
-----
    python3 chartify.py                 # asks for everything, step by step
    python3 chartify.py INPUT [options] # or drive it entirely from flags

Run with no arguments (or -i) and it prompts for the image, fabric count,
finished size, palette, page size and output folder, showing the resulting
dimensions and stitch count as you go. Anything passed as a flag is not
re-asked. The suggested output folder is `<image>-chart-<date>-<time>`, so
repeat runs never overwrite each other.

Examples
--------
    # 400x314 stitches (the "Still Life With A View" size), 5x5-square pages
    python3 chartify.py "still life.jpg" --stitches 400x314 --colors 90

    # Fit to a 14-count aida piece 10 inches wide, cap the palette at 40
    python3 chartify.py photo.png --width-inches 10 --count 14 --colors 40

    # Smaller, less daunting pages: 3x3 big squares each
    python3 chartify.py photo.png --width-stitches 140 --squares-per-page 3

    # Match a kit whose printed sheets are 6 squares across by 8 down
    python3 chartify.py photo.png --stitches 400x314 --squares-per-page 6x8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")

try:
    from dmc_colors import DMC as DMC_LOOKUP, dmc_items
except ImportError:
    sys.exit("dmc_colors.py must sit beside this script")


# --------------------------------------------------------------------------
# Symbol alphabet
# --------------------------------------------------------------------------

# Ordered roughly by legibility at small print sizes: plain capitals first,
# then digits and lowercase, then the distinctive extended glyphs that
# commercial charts fall back on, then a second tier of Greek letters,
# math/dingbat symbols, and pictographs for palettes larger than that.
# Every glyph here is verified to render in DejaVu Sans Mono; deliberately
# excludes whole families (box-drawing pieces, filled/hollow circle or
# triangle sets, dice faces) that only differ by rotation, fill, or dot
# count -- those read as the same symbol at the size a stitch cell prints.
SYMBOLS = (
    "ABCDEFGHJKLMNOPRSTUVWXYZ"
    "0123456789"
    "abcdefhkmnrsuvwxz"
    "Ø‡½«»®Æ±¶Œ"
    "£¤§©¢€ƒ™œ•"
    "!\"#$%&()*+<=>?@^{}~—"
    "αβγδεζηθικλμνξοπρστυφχψω"
    "ΓΔΘΛΞΠΣΦΨΩ"
    "∅∞∠√∝∫∑"
    "▲◆●■★♠♣♥♦♪✓✗✚✱"
    "¡¿×÷¬°µ¦†‣"
    "☀☂☎☕☘⌂⌘"
)

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Pick the first available font that covers the symbol alphabet."""
    for path in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            font = ImageFont.truetype(path, size)
        except OSError:
            continue
        if _covers_alphabet(path):
            return font
    print("  ! no font with full glyph coverage found; falling back to default",
          file=sys.stderr)
    return ImageFont.load_default()


def _covers_alphabet(path: str) -> bool:
    """True if the font has a glyph for every symbol we might emit."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return True  # can't verify; assume the curated candidates are fine
    try:
        cmap = TTFont(path).getBestCmap()
    except Exception:
        return True
    return all(ord(ch) in cmap for ch in SYMBOLS)


# --------------------------------------------------------------------------
# Colour science
# --------------------------------------------------------------------------

def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0-255, shape (...,3)) to CIELAB under D65."""
    srgb = np.asarray(rgb, dtype=np.float64) / 255.0

    # Undo the sRGB transfer function.
    linear = np.where(srgb <= 0.04045,
                      srgb / 12.92,
                      ((srgb + 0.055) / 1.055) ** 2.4)

    # Linear sRGB -> XYZ (D65).
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ m.T

    # Normalise by the D65 white point.
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / white

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([
        116.0 * fy - 16.0,
        500.0 * (fx - fy),
        200.0 * (fy - fz),
    ], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference.

    lab1 is (N,3), lab2 is (M,3); returns an (N,M) distance matrix. This is
    noticeably better than CIE76 at picking the thread a stitcher would agree
    with, particularly across the muted greens and browns that dominate
    landscape designs.
    """
    l1 = lab1[:, None, 0]
    a1 = lab1[:, None, 1]
    b1 = lab1[:, None, 2]
    l2 = lab2[None, :, 0]
    a2 = lab2[None, :, 1]
    b2 = lab2[None, :, 2]

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0

    g = 0.5 * (1.0 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7 + 1e-12)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2

    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dlp = l2 - l1
    dcp = c2p - c1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(c1p * c2p == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp) / 2.0)

    lp_bar = (l1 + l2) / 2.0
    cp_bar = (c1p + c2p) / 2.0

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hp_bar = np.where(
        c1p * c2p == 0.0, hsum,
        np.where(hdiff <= 180.0, hsum / 2.0,
                 np.where(hsum < 360.0, (hsum + 360.0) / 2.0,
                          (hsum - 360.0) / 2.0)))

    t = (1.0
         - 0.17 * np.cos(np.radians(hp_bar - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hp_bar))
         + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0)))

    dtheta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    rc = 2.0 * np.sqrt(cp_bar**7 / (cp_bar**7 + 25.0**7 + 1e-12))
    rt = -rc * np.sin(np.radians(2.0 * dtheta))

    sl = 1.0 + (0.015 * (lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (lp_bar - 50.0) ** 2)
    sc = 1.0 + 0.045 * cp_bar
    sh = 1.0 + 0.015 * cp_bar * t

    return np.sqrt(
        (dlp / sl) ** 2
        + (dcp / sc) ** 2
        + (dHp / sh) ** 2
        + rt * (dcp / sc) * (dHp / sh)
    )


# --------------------------------------------------------------------------
# Palette selection
# --------------------------------------------------------------------------

@dataclass
class Thread:
    code: str
    name: str
    rgb: tuple[int, int, int]
    symbol: str = ""
    stitches: int = 0
    # Fabric count this thread's skein estimate assumes; set per run, because
    # thread consumed per stitch scales with how fine the fabric is.
    count: int = 14

    @property
    def skeins(self) -> int:
        """Skeins of 6-strand floss needed, stitching with 2 strands.

        Anchored on the standard rule of thumb -- about **1,800 full crosses
        per skein on 14-count** -- rather than on thread geometry, because a
        first-principles length calculation comes out roughly 1.7x optimistic:
        it misses the tail wasted at every thread start and finish, which
        dominates in a chart like this where colours change constantly.

        Thread per stitch scales as 1/count, so the stitches a skein covers
        scales with the count: ~1,414 at 11-count, 1,800 at 14, ~2,057 at 16,
        ~2,314 at 18. Deliberately rounded up per colour, and a colour never
        reports zero -- you cannot buy a fraction of a skein.
        """
        per_skein = 1800.0 * (max(1, self.count) / 14.0)
        return max(1, math.ceil(self.stitches / per_skein))

    @property
    def dmc_sort_key(self) -> tuple[int, str]:
        """Numeric DMC order, e.g. room 3799 before 3800 before B5200.

        Plain string sort would put "310" before "45" (lexical, not
        numeric); most DMC codes are plain integers, but a few -- B5200,
        White, Ecru -- aren't, so those fall back to sorting alphabetically
        after every numbered colour rather than raising.
        """
        try:
            return (0, f"{int(self.code):05d}")
        except ValueError:
            return (1, self.code)


def load_palette_csv(path: Path) -> list[Thread]:
    """Read a fixed symbol -> DMC palette from CSV.

    Columns: symbol, dmc, name (optional), skeins (optional). The symbol
    column is authoritative -- this is how you reproduce an existing chart's
    notation instead of letting the script invent its own.
    """
    threads: list[Thread] = []
    seen: set[str] = set()
    with open(path, newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            sym = (row.get("symbol") or "").strip()
            code = (row.get("dmc") or "").strip()
            if not sym or not code:
                continue
            if code not in DMC_LOOKUP:
                print(f"  ! {path.name} line {n}: DMC {code} not in the colour "
                      f"table; skipping", file=sys.stderr)
                continue
            if sym in seen:
                sys.exit(f"{path.name} line {n}: symbol {sym!r} used twice")
            seen.add(sym)
            name, rgb = DMC_LOOKUP[code]
            threads.append(Thread(code, row.get("name", "").strip() or name, rgb,
                                  symbol=sym))
    if not threads:
        sys.exit(f"no usable rows in {path}")
    return threads


def choose_palette(pixels_lab: np.ndarray, max_colors: int):
    """Pick the DMC subset that best covers the image.

    Greedy furthest-point-style reduction: assign every pixel to its nearest
    DMC thread, then keep the threads that actually carry meaningful area,
    dropping the long tail of near-duplicates that would make the chart
    miserable to stitch.
    """
    table = dmc_items()
    dmc_rgb = np.array([rgb for _, _, rgb in table], dtype=np.float64)
    dmc_lab = srgb_to_lab(dmc_rgb)

    # Nearest thread for every distinct pixel colour, in chunks to bound memory.
    nearest = np.empty(len(pixels_lab), dtype=np.int32)
    chunk = 4096
    for start in range(0, len(pixels_lab), chunk):
        block = pixels_lab[start:start + chunk]
        nearest[start:start + chunk] = np.argmin(ciede2000(block, dmc_lab), axis=1)

    counts = np.bincount(nearest, minlength=len(table))
    used = np.argsort(counts)[::-1]
    used = [i for i in used if counts[i] > 0]

    if len(used) <= max_colors:
        keep = used
    else:
        keep = used[:max_colors]

    return [Thread(table[i][0], table[i][1], table[i][2]) for i in keep], dmc_lab, table


def map_to_palette(pixels_lab: np.ndarray, palette: list[Thread]) -> np.ndarray:
    """Assign each pixel to an index into `palette`."""
    pal_lab = srgb_to_lab(np.array([t.rgb for t in palette], dtype=np.float64))
    out = np.empty(len(pixels_lab), dtype=np.int32)
    chunk = 4096
    for start in range(0, len(pixels_lab), chunk):
        block = pixels_lab[start:start + chunk]
        out[start:start + chunk] = np.argmin(ciede2000(block, pal_lab), axis=1)
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_tile(matrix: np.ndarray, palette: list[Thread], *,
                x0: int, y0: int, cell: int, block: int,
                label: str, confirmed: np.ndarray | None = None,
                page_w: int | None = None,
                page_h: int | None = None,
                neighbours: tuple[int, int, int, int] | None = None,
                prefix: str = "") -> Image.Image:
    """Draw one chart page.

    `matrix` is the full symbol-index grid; this draws the window starting at
    (x0, y0). Thin lines separate stitches, bold lines every `block` stitches
    mark the counting squares, and stitch numbers run along the top and left.

    `confirmed`, if given, is a boolean grid the same shape as `matrix` marking
    cells read from a real chart. Unconfirmed cells are drawn grey on a tinted
    ground so a guess can never be mistaken for a transcription -- on a
    reconstruction most cells are predictions, and stitching one is a decision
    the user has to make knowingly.
    """
    # Window actually drawn on this page; edge pages come out short.
    # page_w/page_h let a caller match someone else's pagination (the printed
    # kit chart uses 60x80 pages, not this tool's square default).
    pw = PAGE_W if page_w is None else page_w
    ph = PAGE_H if page_h is None else page_h
    win = matrix[y0:y0 + ph, x0:x0 + pw]
    nrows, ncols = win.shape
    conf_win = (confirmed[y0:y0 + ph, x0:x0 + pw]
                if confirmed is not None else None)

    # With context on, the sheet grows a band on every side carrying a faded
    # colour strip of the adjoining page's abutting stitches, so you can see
    # what the work either side of a seam should look like without unfolding
    # the neighbouring sheet.
    # Band = CONTEXT_CELLS neighbour cells, a half-cell gap, and room for the
    # neighbour's name alongside.
    ctx = (cell * CONTEXT_CELLS + cell // 2 + cell) if neighbours else 0
    margin = cell * 2 + ctx
    # One line for the page/range, plus one per optional note.
    footer = cell * 2 + (cell if conf_win is not None else 0) \
        + (cell if neighbours else 0)
    width = margin + ncols * cell + cell + ctx
    height = margin + nrows * cell + cell + ctx + footer

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    sym_font = load_font(int(cell * 0.68))
    num_font = load_font(max(8, int(cell * 0.45)))

    # Cells and symbols.
    for r in range(nrows):
        for c in range(ncols):
            idx = int(win[r, c])
            thread = palette[idx]
            x = margin + c * cell
            y = margin + r * cell
            is_conf = True if conf_win is None else bool(conf_win[r, c])
            draw.rectangle([x, y, x + cell, y + cell],
                           fill="white" if is_conf else (242, 242, 242))
            sym = thread.symbol
            bbox = draw.textbbox((0, 0), sym, font=sym_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x + (cell - tw) / 2 - bbox[0],
                       y + (cell - th) / 2 - bbox[1]),
                      sym, fill="black" if is_conf else (140, 140, 140),
                      font=sym_font)

    # Thin gridlines.
    for c in range(ncols + 1):
        x = margin + c * cell
        draw.line([x, margin, x, margin + nrows * cell], fill=(170, 170, 170), width=1)
    for r in range(nrows + 1):
        y = margin + r * cell
        draw.line([margin, y, margin + ncols * cell, y], fill=(170, 170, 170), width=1)

    # Bold every `block` stitches, counted in absolute chart coordinates so the
    # heavy lines line up across pages.
    bold = max(2, cell // 8)
    for c in range(ncols + 1):
        if (x0 + c) % block == 0:
            x = margin + c * cell
            draw.line([x, margin, x, margin + nrows * cell], fill="black", width=bold)
    for r in range(nrows + 1):
        if (y0 + r) % block == 0:
            y = margin + r * cell
            draw.line([margin, y, margin + ncols * cell, y], fill="black", width=bold)

    # Outer border.
    draw.rectangle([margin, margin, margin + ncols * cell, margin + nrows * cell],
                   outline="black", width=bold)

    # Stitch numbers every `block` along the top and left edge. Cells are
    # numbered from 1, so the line at absolute index N heads cell N+1.
    for c in range(ncols):
        if (x0 + c) % block == 0:
            x = margin + c * cell
            draw.text((x + 2, margin - cell * 0.9), str(x0 + c + 1),
                      fill="black", font=num_font)
    for r in range(nrows):
        if (y0 + r) % block == 0:
            y = margin + r * cell
            draw.text((4, y + 2), str(y0 + r + 1), fill="black", font=num_font)

    # Context bands: the first CONTEXT_CELLS stitches of each adjoining page,
    # drawn as real symbol cells so you can match notation across the seam.
    # Greyed and set off by a gap, so they read as reference rather than as
    # part of this page -- you should never be in doubt about what to stitch.
    if neighbours:
        px, py, npx, npy = neighbours
        full_h, full_w = matrix.shape
        strip = CONTEXT_CELLS
        ink = (145, 145, 145)
        line = (205, 205, 205)

        def ctx_cell(idx: int, xx: float, yy: float) -> None:
            draw.rectangle([xx, yy, xx + cell, yy + cell],
                           fill=(250, 250, 250), outline=line, width=1)
            sym = palette[int(idx)].symbol
            bb = draw.textbbox((0, 0), sym, font=sym_font)
            draw.text((xx + (cell - (bb[2] - bb[0])) / 2 - bb[0],
                       yy + (cell - (bb[3] - bb[1])) / 2 - bb[1]),
                      sym, fill=ink, font=sym_font)

        # The gap that separates neighbour cells from this page's grid.
        gap = cell // 2
        grid_r = margin + ncols * cell      # right edge of this page's grid
        grid_b = margin + nrows * cell      # bottom edge

        # Left neighbour: its right-hand columns, abutting our column 1.
        if x0 > 0:
            src = matrix[y0:y0 + nrows, max(0, x0 - strip):x0]
            for r in range(src.shape[0]):
                for c in range(src.shape[1]):
                    ctx_cell(src[r, c],
                             margin - gap - (src.shape[1] - c) * cell,
                             margin + r * cell)
        # Right neighbour: its left-hand columns.
        if x0 + ncols < full_w:
            src = matrix[y0:y0 + nrows, x0 + ncols:min(full_w, x0 + ncols + strip)]
            for r in range(src.shape[0]):
                for c in range(src.shape[1]):
                    ctx_cell(src[r, c], grid_r + gap + c * cell,
                             margin + r * cell)
        # Top neighbour: its bottom rows.
        if y0 > 0:
            src = matrix[max(0, y0 - strip):y0, x0:x0 + ncols]
            for r in range(src.shape[0]):
                for c in range(src.shape[1]):
                    ctx_cell(src[r, c], margin + c * cell,
                             margin - gap - (src.shape[0] - r) * cell)
        # Bottom neighbour: its top rows.
        if y0 + nrows < full_h:
            src = matrix[y0 + nrows:min(full_h, y0 + nrows + strip), x0:x0 + ncols]
            for r in range(src.shape[0]):
                for c in range(src.shape[1]):
                    ctx_cell(src[r, c], margin + c * cell,
                             grid_b + gap + r * cell)

        # Name the adjoining sheets in their bands, so the seam tells you which
        # page to pick up next.
        def nb_label(text: str, cx: float, cy: float) -> None:
            bb = draw.textbbox((0, 0), text, font=num_font)
            draw.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2),
                      text, fill=(110, 110, 110), font=num_font)

        mid_y = margin + nrows * cell / 2
        mid_x = margin + ncols * cell / 2
        # Side labels sit inside the band and are rotated, so a long page name
        # can't spill off the sheet edge the way horizontal text does.
        def side_label(text: str, cx: float, cy: float) -> None:
            bb = draw.textbbox((0, 0), text, font=num_font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            tag = Image.new("RGB", (tw + 4, th + 4), "white")
            ImageDraw.Draw(tag).text((2 - bb[0], 2 - bb[1]), text,
                                     fill=(110, 110, 110), font=num_font)
            tag = tag.rotate(90, expand=True)
            img.paste(tag, (int(cx - tag.width / 2), int(cy - tag.height / 2)))

        # Labels sit beyond the context cells, in the outer part of the band.
        band = cell * CONTEXT_CELLS + gap
        if x0 > 0:
            side_label(f"{prefix}x{px - 1}y{py}",
                       margin - band - cell / 2, mid_y)
        if x0 + ncols < full_w:
            side_label(f"{prefix}x{px + 1}y{py}",
                       grid_r + band + cell / 2, mid_y)
        if y0 > 0:
            nb_label(f"{prefix}x{px}y{py - 1}", mid_x, margin - band - cell / 2)
        if y0 + nrows < full_h:
            nb_label(f"{prefix}x{px}y{py + 1}", mid_x, grid_b + band + cell / 2)

        # Locator map: the whole chart as a page grid, this page filled. Placed
        # in the footer strip, clear of both the chart and the context bands.
        mw = min(ctx * 2, cell * 5)
        box = max(4, int(mw / max(npx, npy)))
        mx = width - box * npx - cell // 2
        my = height - box * npy - cell // 2
        for gy in range(npy):
            for gx in range(npx):
                x1, y1 = mx + gx * box, my + gy * box
                here = (gx + 1 == px and gy + 1 == py)
                draw.rectangle([x1, y1, x1 + box, y1 + box],
                               fill=(70, 70, 70) if here else (235, 235, 235),
                               outline=(150, 150, 150))

    # Footer: which page this is and which stitches it covers. Sits below the
    # bottom context band, not on top of it.
    foot_y = margin + nrows * cell + ctx + cell * 0.4
    foot = f"{label}   columns {x0 + 1}-{x0 + ncols}   rows {y0 + 1}-{y0 + nrows}"
    draw.text((margin, foot_y), foot, fill="black", font=num_font)
    note_y = foot_y + cell * 0.7
    if conf_win is not None:
        # Second line: the footer has to fit the page width, so don't append.
        n_conf = int(conf_win.sum())
        draw.text((margin, note_y),
                  f"{n_conf}/{nrows * ncols} cells read from the chart; "
                  f"grey symbols on tinted cells are PREDICTED, not read",
                  fill=(90, 90, 90), font=num_font)
        note_y += cell * 0.7
    if neighbours:
        draw.text((margin, note_y),
                  f"grey edge cells = first {CONTEXT_CELLS} stitches of the "
                  f"adjoining page, for matching across the seam; "
                  f"they are NOT part of this page",
                  fill=(120, 120, 120), font=num_font)
    return img


def write_pdf_streaming(pages: list[Path], out: Path, *, dpi: int,
                        quality: int = 85) -> None:
    """Write a multi-page PDF one page at a time.

    Pillow's own `save(save_all=True, append_images=...)` decodes and holds
    every page for the whole write: 40 Letter sheets at 300dpi peaked at
    ~1.3GB here, which overruns the heap in a memory-capped environment and
    surfaces as `MemoryError` mid-save. Neither spooling the pages to disk
    nor handing it pre-encoded JPEGs avoids that -- it decodes them anyway.

    So the container is written directly. Each page becomes a JPEG-compressed
    /DCTDecode image XObject, encoded and released before the next page is
    read, which holds peak memory to a single page (~87MB for the same 40
    sheets). The structure is deliberately plain: one Page plus one Image
    plus one content stream per sheet, and a classic xref table.

    `dpi` sets the physical page size: a 2550x3300 pixel sheet at 300dpi
    becomes 612x792pt, i.e. Letter.
    """
    import io

    offsets: dict[int, int] = {}

    with open(out, "wb") as fh:
        # The binary comment marks the file as containing 8-bit data.
        fh.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        def emit(num: int, body: bytes) -> None:
            offsets[num] = fh.tell()
            fh.write(f"{num} 0 obj\n".encode("ascii"))
            fh.write(body)
            fh.write(b"\nendobj\n")

        count = len(pages)
        # Object numbering: 1 Catalog, 2 Pages, then three per sheet
        # (Page, Image, Contents) so each sheet's numbers are predictable.
        page_num = lambda i: 3 + 3 * i          # noqa: E731
        kids = " ".join(f"{page_num(i)} 0 R" for i in range(count))

        emit(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        emit(2, f"<< /Type /Pages /Count {count} /Kids [{kids}] >>"
                .encode("ascii"))

        for i, path in enumerate(pages):
            with Image.open(path) as src:
                page = src.convert("RGB")
                width, height = page.size
                buf = io.BytesIO()
                page.save(buf, "JPEG", quality=quality)
                page.close()
            jpeg = buf.getvalue()
            buf.close()

            pt_w = width * 72.0 / dpi
            pt_h = height * 72.0 / dpi
            p_n = page_num(i)
            img_n, cont_n = p_n + 1, p_n + 2

            emit(p_n, (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {pt_w:.2f} {pt_h:.2f}] "
                f"/Resources << /XObject << /Im0 {img_n} 0 R >> "
                f"/ProcSet [/PDF /ImageC] >> "
                f"/Contents {cont_n} 0 R >>").encode("ascii"))

            # The image stream is written straight out rather than built as
            # one bytes object, so the JPEG is the only page-sized thing
            # resident.
            offsets[img_n] = fh.tell()
            fh.write(f"{img_n} 0 obj\n".encode("ascii"))
            fh.write((
                f"<< /Type /XObject /Subtype /Image /Width {width} "
                f"/Height {height} /ColorSpace /DeviceRGB "
                f"/BitsPerComponent 8 /Filter /DCTDecode "
                f"/Length {len(jpeg)} >>\nstream\n").encode("ascii"))
            fh.write(jpeg)
            fh.write(b"\nendstream\nendobj\n")
            del jpeg

            # Scale the unit image up to the full page box.
            content = (f"q {pt_w:.2f} 0 0 {pt_h:.2f} 0 0 cm /Im0 Do Q"
                       .encode("ascii"))
            emit(cont_n, b"<< /Length %d >>\nstream\n" % len(content)
                 + content + b"\nendstream")

        start_xref = fh.tell()
        highest = max(offsets)
        fh.write(f"xref\n0 {highest + 1}\n".encode("ascii"))
        fh.write(b"0000000000 65535 f \n")
        for num in range(1, highest + 1):
            if num in offsets:
                fh.write(f"{offsets[num]:010d} 00000 n \n".encode("ascii"))
            else:
                fh.write(b"0000000000 00000 f \n")
        fh.write((f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\n"
                  f"startxref\n{start_xref}\n%%EOF\n").encode("ascii"))


def add_binder_margin(img: Image.Image, *, inches: float, dpi: int) -> Image.Image:
    """Pad a blank strip onto the left edge for hole-punching.

    Widening the page rather than shrinking the chart into it, so the margin
    never eats into stitch cells that are already sized for legibility.
    """
    if inches <= 0:
        return img
    strip = round(inches * dpi)
    padded = Image.new("RGB", (img.width + strip, img.height), "white")
    padded.paste(img, (strip, 0))
    return padded


def fit_to_page(img: Image.Image, *, page_w_in: float, page_h_in: float,
               margin_in: float, dpi: int) -> Image.Image:
    """Centre an already-margined sheet on a fixed physical page.

    Every page comes out the same `page_w_in`x`page_h_in` size (e.g.
    Letter) regardless of the content's own shape -- a tile is roughly
    square, a cover is a wide thumbnail-and-list layout, a leftover tail
    column can be only a couple of stitches wide. Letting the page shrink
    to match each of those made most of them a different odd size, which
    is worse than some blank space: a booklet where every sheet really is
    the same paper size, with the chart centred on it, prints predictably
    and looks intentional even when a page is mostly blank.

    Content is scaled up to fill one axis fully (no distortion), then
    scaled down instead if it would still overflow the page.

    The binder margin is already on the left of `img` and is kept as its
    own space to the left of the centred content, so the punched edge
    lines up the same way on every sheet regardless of how much blank
    space that particular page has.
    """
    page_w = round(page_w_in * dpi)
    page_h = round(page_h_in * dpi)
    left_pad = round(margin_in * dpi)
    edge_pad = round(0.25 * dpi)  # top/right/bottom safety margin

    usable_w = max(1, page_w - left_pad - edge_pad)
    usable_h = max(1, page_h - 2 * edge_pad)
    scale = min(usable_w / img.width, usable_h / img.height)
    new_w = max(1, round(img.width * scale))
    new_h = max(1, round(img.height * scale))
    if scale != 1.0:
        img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (page_w, page_h), "white")
    x = left_pad + (usable_w - new_w) // 2
    y = edge_pad + (usable_h - new_h) // 2
    canvas.paste(img, (x, y))
    return canvas


def render_key(palette: list[Thread], cell: int = 28) -> Image.Image:
    """Draw the symbol -> DMC colour key."""
    rows = len(palette)
    per_col = math.ceil(rows / 2) if rows > 30 else rows
    ncols = math.ceil(rows / per_col)

    col_w = cell * 24
    width = col_w * ncols + cell
    height = cell * (per_col + 3)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(int(cell * 0.55))
    head = load_font(int(cell * 0.6))

    draw.text((cell // 2, cell // 2), "Thread key", fill="black", font=head)

    for n, thread in enumerate(palette):
        col = n // per_col
        row = n % per_col
        x = cell // 2 + col * col_w
        y = cell * 2 + row * cell

        # Symbol cell.
        draw.rectangle([x, y, x + cell, y + cell], outline="black", width=1)
        bbox = draw.textbbox((0, 0), thread.symbol, font=font)
        draw.text((x + (cell - (bbox[2] - bbox[0])) / 2 - bbox[0],
                   y + (cell - (bbox[3] - bbox[1])) / 2 - bbox[1]),
                  thread.symbol, fill="black", font=font)

        # Colour swatch.
        sx = x + cell * 1.3
        draw.rectangle([sx, y, sx + cell, y + cell], fill=thread.rgb,
                       outline="black", width=1)

        # Text.
        draw.text((sx + cell * 1.4, y + cell * 0.25),
                  f"DMC {thread.code:<6} {thread.name}"
                  f"   ({thread.stitches} st, ~{thread.skeins} skein"
                  f"{'s' if thread.skeins != 1 else ''})",
                  fill="black", font=font)

    return img


def render_shopping_list(palette: list[Thread], cell: int = 28) -> Image.Image:
    """DMC-numbered thread list, for buying floss at the store.

    The cover's list is sorted by how much of each colour the design
    uses, which is the right order for deciding what matters -- but a
    shelf of DMC floss is racked by number, not by usage, so a page
    sorted the way the rack is organised means fewer trips back and
    forth checking the list.
    """
    used = sorted((t for t in palette if t.stitches), key=lambda t: t.dmc_sort_key)
    rows = len(used)
    per_col = math.ceil(rows / 2) if rows > 30 else rows
    ncols = math.ceil(rows / per_col)

    col_w = cell * 24
    width = col_w * ncols + cell
    height = cell * (per_col + 3)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(int(cell * 0.55))
    head = load_font(int(cell * 0.6))

    total_skeins = sum(t.skeins for t in used)
    draw.text((cell // 2, cell // 2),
              f"Shopping list -- DMC order, {total_skeins} skeins total",
              fill="black", font=head)

    for n, thread in enumerate(used):
        col = n // per_col
        row = n % per_col
        x = cell // 2 + col * col_w
        y = cell * 2 + row * cell

        draw.rectangle([x, y, x + cell, y + cell], fill=thread.rgb,
                       outline="black", width=1)

        sk = f"{thread.skeins} skein{'s' if thread.skeins != 1 else ''}"
        draw.text((x + cell * 1.3, y + cell * 0.25),
                  f"DMC {thread.code:<6} {thread.name:<24} "
                  f"{thread.stitches:>6,} st   {sk}",
                  fill="black", font=font)

    return img


def render_preview(matrix: np.ndarray, palette: list[Thread], scale: int = 4) -> Image.Image:
    """Flat colour preview of the finished piece."""
    rows, cols = matrix.shape
    lut = np.array([t.rgb for t in palette], dtype=np.uint8)
    rgb = lut[matrix]
    return Image.fromarray(rgb, "RGB").resize((cols * scale, rows * scale), Image.NEAREST)


def render_preview_page(matrix: np.ndarray, palette: list[Thread], *,
                        source: Path, width: int = 1400) -> Image.Image:
    """A titled page showing what the finished piece will look like.

    The chart pages are notation, not a picture -- this is the only sheet
    in the booklet that shows the actual colours, so it's what you'd hand
    someone to ask "does this look right?" before committing thread to
    fabric.
    """
    rows, cols = matrix.shape
    preview = render_preview(matrix, palette, scale=1)
    pad = 40
    max_w, max_h = width - pad * 2, 1600
    scale = min(max_w / cols, max_h / rows)
    preview = preview.resize((max(1, round(cols * scale)),
                              max(1, round(rows * scale))), Image.NEAREST)

    h1 = load_font(34)
    h2 = load_font(20)
    header_h = 44 + 34 + 22
    height = pad + header_h + preview.height + pad

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = pad
    draw.text((pad, y), source.stem, fill="black", font=h1)
    y += 44
    draw.text((pad, y), "finished preview", fill=(110, 110, 110), font=h2)
    y += 34
    draw.line([pad, y, width - pad, y], fill=(200, 200, 200), width=2)
    y += 22

    img.paste(preview, ((width - preview.width) // 2, y))
    return img


def render_cover(matrix: np.ndarray, palette: list[Thread], *,
                 source: Path, count: int, page_w: int, page_h: int,
                 block: int, prefix: str, palette_file: Path | None,
                 learn_file: Path | None, confirmed: np.ndarray | None,
                 width: int = 1400) -> Image.Image:
    """Cover sheet: the settings this chart was built with, plus a shopping list.

    Revisiting a project months later, the settings are the thing you cannot
    recover by looking at the pages -- fabric count and finished size in
    particular are invisible in the notation itself.
    """
    rows, cols = matrix.shape
    npx, npy = math.ceil(cols / page_w), math.ceil(rows / page_h)
    used = [t for t in palette if t.stitches]
    total_skeins = sum(t.skeins for t in used)

    h1 = load_font(34)
    h2 = load_font(20)
    body = load_font(16)
    small = load_font(13)

    # Thumbnail of the finished piece: a third of the sheet wide, but height
    # capped so a tall portrait image doesn't stretch the whole cover.
    thumb = render_preview(matrix, palette, scale=1)
    tw, th_cap = width // 3, 420
    tw = min(tw, max(1, round(th_cap * cols / rows)))
    thumb = thumb.resize((tw, max(1, round(tw * rows / cols))), Image.LANCZOS)

    pad = 40
    # Two-column thread list. Build the fact rows first so the sheet is sized
    # from what is actually drawn rather than from a guess.
    per_col = max(1, math.ceil(len(used) / 2))
    list_h = per_col * 22

    w_in, h_in = cols / count, rows / count
    facts = [
        ("source image", source.name),
        ("stitch grid", f"{cols} x {rows}  ({cols * rows:,} stitches)"),
        ("fabric count", f"{count}-count  ({count} stitches per inch)"),
        ("finished size",
         f'{w_in:.1f}" x {h_in:.1f}"   ({w_in / 12:.2f}ft x {h_in / 12:.2f}ft)'),
        ("fabric needed",
         f'{w_in + 6:.0f}" x {h_in + 6:.0f}" (3" margin all round)'),
        ("thread colours", f"{len(used)} DMC colours"),
        ("floss needed", f"~{total_skeins} skeins (2 strands)"),
        ("pages", f"{npx} across x {npy} down = {npx * npy} sheets"),
        ("page size", f"{page_w} x {page_h} stitches "
                      f"({page_w // block}x{page_h // block} squares of {block})"),
        ("sheet files", f"{prefix}page01 ... {prefix}page{npx * npy:02d}"),
    ]
    if palette_file:
        facts.append(("symbol key", f"{palette_file.name} (fixed)"))
    else:
        facts.append(("symbol key", "chosen automatically by area"))
    if learn_file:
        facts.append(("learned from", learn_file.name))
    if confirmed is not None:
        n = int(confirmed.sum())
        facts.append(("cells read from chart",
                      f"{n:,} of {rows * cols:,} ({100.0 * n / (rows * cols):.1f}%)"
                      f" -- the rest predicted"))

    # header + rule + (facts beside thumbnail) + rule + list heading + list
    header_h = 44 + 34 + 22
    mid_h = max(thumb.height, len(facts) * 23)
    height = pad + header_h + mid_h + 26 + 20 + 30 + 24 + list_h + pad

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = pad

    draw.text((pad, y), source.stem, fill="black", font=h1)
    y += 44
    draw.text((pad, y), "cross-stitch chart", fill=(110, 110, 110), font=h2)
    y += 34
    draw.line([pad, y, width - pad, y], fill=(200, 200, 200), width=2)
    y += 22

    # ---- settings block, beside the thumbnail --------------------------
    img.paste(thumb, (width - pad - thumb.width, y))
    facts_top = y

    label_w = 190
    for k, v in facts:
        draw.text((pad, y), k, fill=(120, 120, 120), font=body)
        draw.text((pad + label_w, y), v, fill="black", font=body)
        y += 23

    y = max(y, facts_top + thumb.height) + 26
    draw.line([pad, y, width - pad, y], fill=(200, 200, 200), width=2)
    y += 20

    # ---- shopping list ------------------------------------------------
    draw.text((pad, y), f"Thread shopping list -- {total_skeins} skeins total",
              fill="black", font=h2)
    y += 30
    per_skein = round(1800 * count / 14)
    draw.text((pad, y),
              f"estimate: ~{per_skein:,} stitches per skein at {count}-count, "
              f"rounded up per colour. Buy a spare of the big ones.",
              fill=(130, 130, 130), font=small)
    y += 24

    col_w = (width - pad * 2) // 2
    order = sorted(used, key=lambda t: -t.stitches)
    for i, t in enumerate(order):
        cx = pad + (i // per_col) * col_w
        cy = y + (i % per_col) * 22
        draw.rectangle([cx, cy + 3, cx + 13, cy + 16], fill=t.rgb,
                       outline=(90, 90, 90))
        draw.text((cx + 20, cy), t.symbol, fill="black", font=body)
        draw.text((cx + 42, cy + 1), f"DMC {t.code}", fill="black", font=small)
        draw.text((cx + 120, cy + 1), t.name[:24], fill=(80, 80, 80), font=small)
        # Right-align both numbers off the column edge so long stitch counts
        # can't run into the skein figure.
        sk = f"{t.skeins} skein{'s' if t.skeins != 1 else ''}"
        bb = draw.textbbox((0, 0), sk, font=small)
        sk_x = cx + col_w - 30 - (bb[2] - bb[0])
        draw.text((sk_x, cy + 1), sk, fill="black", font=small)
        st = f"{t.stitches:,} st"
        bb = draw.textbbox((0, 0), st, font=small)
        draw.text((sk_x - 14 - (bb[2] - bb[0]), cy + 1), st,
                  fill=(80, 80, 80), font=small)
    return img


def render_page_map(matrix: np.ndarray, palette: list[Thread], *,
                    page_w: int, page_h: int, prefix: str,
                    scale: int = 4, target_px: int = 1400) -> Image.Image:
    """The whole design, divided up and labelled with its page names.

    Answers "which sheet covers this part of the picture?" -- the tiled pages
    show notation but not what they depict, so without this you have to guess
    from stitch numbers alone.
    """
    rows, cols = matrix.shape
    # Scale so the long edge lands near target_px, keeping page cuts on exact
    # pixel boundaries: a fractional scale would smear the boundary lines.
    scale = max(2, min(scale if scale > 1 else 4,
                       round(target_px / max(cols, rows)) or 2))
    art = render_preview(matrix, palette, scale=scale)

    npx = math.ceil(cols / page_w)
    npy = math.ceil(rows / page_h)

    pad = 44
    img = Image.new("RGB", (art.width + pad * 2, art.height + pad * 2), "white")
    img.paste(art, (pad, pad))
    draw = ImageDraw.Draw(img, "RGBA")

    # Size the label to the page box it has to sit inside, so a long name can
    # never straddle a boundary and read as belonging to the wrong page.
    box_w = page_w * scale
    probe = load_font(20)
    name_w = draw.textbbox((0, 0), f"{prefix}x{npx}y{npy}", font=probe)[2]
    fit = int(20 * (box_w * 0.82) / max(1, name_w))
    font = load_font(max(9, min(26, fit)))
    small = load_font(max(8, int(font.size * 0.62)))

    # Dim alternating pages like a chequerboard so the division is legible even
    # where the artwork itself is flat and the boundary lines vanish into it.
    for gy in range(npy):
        for gx in range(npx):
            if (gx + gy) % 2:
                continue
            x1 = pad + gx * page_w * scale
            y1 = pad + gy * page_h * scale
            x2 = min(pad + art.width, x1 + page_w * scale)
            y2 = min(pad + art.height, y1 + page_h * scale)
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, 60))

    # Page boundaries.
    for gx in range(npx + 1):
        x = pad + min(gx * page_w, cols) * scale
        draw.line([x, pad, x, pad + art.height], fill=(220, 30, 30), width=3)
    for gy in range(npy + 1):
        y = pad + min(gy * page_h, rows) * scale
        draw.line([pad, y, pad + art.width, y], fill=(220, 30, 30), width=3)

    # Page names, centred on each page, with a halo so they stay readable over
    # both light and dark artwork. Edge pages can be much narrower than a full
    # page, so each label is sized to the box it actually has and dropped to a
    # bare page number when even that won't fit -- an overflowing label would
    # stray onto the neighbouring page and mislabel it.
    for gy in range(npy):
        for gx in range(npx):
            c0, r0 = gx * page_w, gy * page_h
            c1, r1 = min(cols, c0 + page_w), min(rows, r0 + page_h)
            bw_avail = (c1 - c0) * scale
            bh_avail = (r1 - r0) * scale
            cx = pad + (c0 + c1) / 2 * scale
            cy = pad + (r0 + r1) / 2 * scale
            name = f"{prefix}x{gx + 1}y{gy + 1}"
            rng = f"c{c0 + 1}-{c1}  r{r0 + 1}-{r1}"

            f_name, f_rng = font, small
            if draw.textbbox((0, 0), name, font=f_name)[2] > bw_avail - 8:
                f_name = load_font(max(7, int(font.size * 0.6)))
                name = f"x{gx + 1}y{gy + 1}"
            show_rng = draw.textbbox((0, 0), rng, font=f_rng)[2] <= bw_avail - 8
            if draw.textbbox((0, 0), name, font=f_name)[2] > bw_avail - 4:
                continue  # too narrow for any honest label

            nb = draw.textbbox((0, 0), name, font=f_name)
            rb = draw.textbbox((0, 0), rng, font=f_rng) if show_rng else (0, 0, 0, 0)
            bw = min(bw_avail - 2,
                     max(nb[2] - nb[0], rb[2] - rb[0]) + 16)
            bh = min(bh_avail - 2,
                     (nb[3] - nb[1]) + (rb[3] - rb[1] if show_rng else 0) + 16)
            draw.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                           fill=(255, 255, 255, 205), outline=(200, 60, 60))
            draw.text((cx - (nb[2] - nb[0]) / 2 - nb[0],
                       cy - bh / 2 + 6 - nb[1]),
                      name, fill=(150, 20, 20), font=f_name)
            if show_rng:
                draw.text((cx - (rb[2] - rb[0]) / 2 - rb[0],
                           cy + 2 - rb[1]),
                          rng, fill=(70, 70, 70), font=f_rng)

    # Edge rulers: page index along the top, row index down the left.
    for gx in range(npx):
        x = pad + (gx * page_w + min(page_w, cols - gx * page_w) / 2) * scale
        t = f"x{gx + 1}"
        bb = draw.textbbox((0, 0), t, font=small)
        draw.text((x - (bb[2] - bb[0]) / 2, pad - 26), t, fill="black", font=small)
    for gy in range(npy):
        y = pad + (gy * page_h + min(page_h, rows - gy * page_h) / 2) * scale
        t = f"y{gy + 1}"
        bb = draw.textbbox((0, 0), t, font=small)
        draw.text((6, y - (bb[3] - bb[1]) / 2), t, fill="black", font=small)

    draw.text((pad, pad + art.height + 12),
              f"{npx} x {npy} pages of {page_w}x{page_h} stitches "
              f"({cols}x{rows} total)", fill=(60, 60, 60), font=small)
    return img


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

# Stitches per page, across and down. Filled in from args at runtime; pages
# need not be square (a 6x8-square page matches the Still Life kit's sheets).
PAGE_W = 50
PAGE_H = 50

# How many stitches of each adjoining page to show around the edges. Two is
# enough to line up notation across a seam without crowding the sheet.
CONTEXT_CELLS = 2

def learn_from_cells(training: Path, small_rgb: np.ndarray,
                     palette: list[Thread], k: int = 5) -> np.ndarray:
    """Predict every stitch by learning from already-known chart cells.

    `training` is a CSV of col,row,symbol taken from a real chart (decoded
    pages, hand-transcribed blocks). Those cells pin down how this particular
    designer mapped the artwork's colours onto threads, which generic nearest-
    DMC matching cannot recover -- two threads a fraction of a deltaE apart are
    a coin flip without evidence, and the designer's choice is the evidence.

    Cells listed in the training file are copied through verbatim; everything
    else is predicted by k-nearest-neighbour vote in CIELAB.

    Only threads that appear in the training data can ever be predicted, so
    coverage of the design's colour regions matters more than raw cell count.
    """
    sym_to_idx = {t.symbol: n for n, t in enumerate(palette)}
    h, w, _ = small_rgb.shape

    known: dict[tuple[int, int], int] = {}
    feats: list[np.ndarray] = []
    labels: list[int] = []
    unknown_syms: Counter = Counter()

    with open(training, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                x = int(row["col"]) - 1
                y = int(row["row"]) - 1
            except (KeyError, ValueError):
                continue
            sym = row.get("symbol", "")
            if not (0 <= x < w and 0 <= y < h):
                continue
            if sym not in sym_to_idx:
                unknown_syms[sym] += 1
                continue
            idx = sym_to_idx[sym]
            known[(x, y)] = idx
            feats.append(small_rgb[y, x])
            labels.append(idx)

    if not feats:
        sys.exit(f"{training}: no usable training cells (check col/row/symbol "
                 f"columns and that symbols match the palette)")
    if unknown_syms:
        print(f"  ! {sum(unknown_syms.values())} training cells use symbols not "
              f"in the palette, ignored: "
              f"{', '.join(sorted(unknown_syms)[:10])}")

    train_lab = srgb_to_lab(np.array(feats))
    train_y = np.array(labels)
    n_threads = len(palette)
    print(f"training    : {len(feats)} known cells, "
          f"{len(set(labels))}/{n_threads} threads represented")

    # Predict the whole grid by kNN vote, in chunks to bound memory.
    flat = srgb_to_lab(small_rgb.reshape(-1, 3))
    out = np.empty(len(flat), dtype=np.int32)
    kk = min(k, len(train_lab))
    step = 2048
    for start in range(0, len(flat), step):
        block = flat[start:start + step]
        d = np.sqrt(((block[:, None, :] - train_lab[None, :, :]) ** 2).sum(-1))
        nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        for i, cand in enumerate(nn):
            # Distance-weighted vote: closer exemplars count for more.
            votes: dict[int, float] = {}
            for j in cand:
                lbl = int(train_y[j])
                votes[lbl] = votes.get(lbl, 0.0) + 1.0 / (1.0 + d[i, j])
            out[start + i] = max(votes.items(), key=lambda kv: kv[1])[0]

    matrix = out.reshape(h, w)

    # Known cells are ground truth -- never let the model overwrite them.
    for (x, y), idx in known.items():
        matrix[y, x] = idx
    print(f"            : {len(known)} cells copied from training data verbatim")
    return matrix


def confirmed_mask(training: Path, rows: int, cols: int) -> np.ndarray:
    """Boolean grid marking cells that were read off a real chart.

    Mirrors the coordinate handling in `learn_from_cells` -- the CSV is 1-based
    and out-of-range cells are ignored -- so the mask can't drift out of step
    with the cells actually copied through verbatim.
    """
    mask = np.zeros((rows, cols), dtype=bool)
    with open(training, newline="") as fh:
        for rec in csv.DictReader(fh):
            try:
                c = int(rec["col"]) - 1
                r = int(rec["row"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= r < rows and 0 <= c < cols:
                mask[r, c] = True
    return mask


def detect_border(img: Image.Image, threshold: int = 235,
                  coverage: float = 0.6, max_trim: int = 8) -> tuple[int, int, int, int]:
    """Find near-white scan/crop edge rows and columns.

    A 1px white edge is common in scanned or cropped source art, and it does
    real damage: at 400x314 each stitch averages ~2.5 source rows, so a single
    white row is ~40% of the first stitch row and drags it light enough to
    pick the wrong thread across the whole row.

    Returns (left, top, right, bottom) counts to trim.
    """
    a = np.asarray(img.convert("RGB")).astype(int)
    h, w, _ = a.shape

    def whiteish(line: np.ndarray) -> bool:
        return ((line > threshold).all(axis=1)).mean() >= coverage

    top = 0
    while top < min(max_trim, h - 1) and whiteish(a[top]):
        top += 1
    bottom = 0
    while bottom < min(max_trim, h - 1 - top) and whiteish(a[h - 1 - bottom]):
        bottom += 1
    left = 0
    while left < min(max_trim, w - 1) and whiteish(a[:, left]):
        left += 1
    right = 0
    while right < min(max_trim, w - 1 - left) and whiteish(a[:, w - 1 - right]):
        right += 1
    return left, top, right, bottom


def parse_stitches(text: str) -> tuple[int, int]:
    if "x" not in text.lower():
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, e.g. 400x314")
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


# --------------------------------------------------------------------------
# Interactive prompts
# --------------------------------------------------------------------------

def _ask(prompt: str, default: str = "", *,
         validate=None, allow_blank: bool = False) -> str:
    """Ask until the answer validates. Enter alone accepts the default."""
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            got = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            print()
            sys.exit("cancelled")
        except KeyboardInterrupt:
            print()
            sys.exit("cancelled")
        if not got:
            got = default
        if not got and not allow_blank:
            print("  -- a value is needed")
            continue
        if validate:
            problem = validate(got)
            if problem:
                print(f"  -- {problem}")
                continue
        return got


def _ask_choice(prompt: str, options: list[tuple[str, str]],
                default: int = 1) -> str:
    """Numbered menu; returns the chosen option's value."""
    print(f"\n{prompt}")
    for n, (_, desc) in enumerate(options, 1):
        mark = " (default)" if n == default else ""
        print(f"  {n}. {desc}{mark}")

    def check(s: str) -> str | None:
        if not s.isdigit() or not 1 <= int(s) <= len(options):
            return f"enter a number from 1 to {len(options)}"
        return None

    return options[int(_ask("  choice", str(default), validate=check)) - 1][0]


# Interactive-only fields that should be remembered for a re-run. Excludes
# things tied to a single run (input path, outdir, prefix, the interactive
# flag itself) so re-running always gets a fresh output folder.
_SAVED_SETTINGS = (
    "count", "stitches", "width_stitches", "width_inches",
    "colors", "palette", "squares_per_page", "margin_inches",
    "format", "no_context",
)


def _settings_path(image: Path) -> Path:
    return image.with_suffix("").with_suffix(".chartify.json")


def _save_settings(image: Path, a: argparse.Namespace) -> None:
    data = {}
    for name in _SAVED_SETTINGS:
        val = getattr(a, name)
        data[name] = str(val) if isinstance(val, Path) else val
    try:
        _settings_path(image).write_text(json.dumps(data, indent=2) + "\n")
    except OSError as e:
        print(f"  -- couldn't save settings: {e}")


def _load_settings(image: Path) -> dict | None:
    path = _settings_path(image)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _suggest_outdir(image: Path) -> str:
    """`<image-stem>-chart-<YYYYmmdd-HHMM>`, so runs never clobber each other."""
    from datetime import datetime
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "-"
                   for ch in image.stem).strip("-").lower() or "chart"
    return f"{stem}-chart-{datetime.now().strftime('%Y%m%d-%H%M')}"


def interactive(argv_defaults: argparse.Namespace) -> argparse.Namespace:
    """Walk the user through the options and return filled-in args.

    Only asks what actually changes the output; anything already given on the
    command line is respected and not re-asked.
    """
    a = argv_defaults
    print("\nchartify -- image to cross-stitch chart")
    print("-" * 52)
    print("Enter alone accepts the default in [brackets]. Ctrl-C to quit.")

    # ---- source image -------------------------------------------------
    if not a.input:
        def has_image(s: str) -> str | None:
            p = Path(s).expanduser()
            if not p.exists():
                return f"no such file: {p}"
            if p.is_dir():
                return "that's a directory, not an image"
            try:
                with Image.open(p):
                    pass
            except Exception:
                return "not a readable image"
            return None

        here = sorted(p for p in Path().glob("*")
                      if p.suffix.lower() in
                      (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"))
        if here:
            print("\nimages in this folder:")
            for p in here[:12]:
                mark = " (has saved settings)" if _settings_path(p).exists() else ""
                print(f"  {p.name}{mark}")
            if len(here) > 12:
                print(f"  ... and {len(here) - 12} more")
        a.input = _ask("\nsource image", validate=has_image)

    src = Path(a.input).expanduser()
    with Image.open(src) as probe:
        iw, ih = probe.size
    aspect = ih / iw
    print(f"\n{src.name}: {iw}x{ih}px, aspect {aspect:.3f} "
          f"({'portrait' if aspect > 1 else 'landscape' if aspect < 1 else 'square'})")

    # ---- reuse saved settings ------------------------------------------
    saved = _load_settings(src)
    reused = False
    if saved and _ask_choice(
            f"found saved settings from a previous run on {src.name}",
            [("reuse", "use them again (you can still change anything below)"),
             ("fresh", "ignore them and start over")],
            default=1) == "reuse":
        reused = True
        for name in _SAVED_SETTINGS:
            if name not in saved:
                continue
            val = saved[name]
            if name == "palette" and val is not None:
                val = Path(val)
            elif name == "stitches" and val is not None:
                val = tuple(val)
            setattr(a, name, val)
        print("  -- reused; enter alone at any prompt below keeps the saved value")

    # ---- fabric count -------------------------------------------------
    if not reused and a.count == 14 and not a.stitches and not a.width_stitches:
        a.count = int(_ask_choice(
            "fabric count (stitches per inch)",
            [("14", "14-count aida -- easiest on the eyes, fewest stitches"),
             ("16", "16-count -- finer detail, moderate effort"),
             ("18", "18-count -- finest detail, most stitches"),
             ("11", "11-count -- very coarse, good for beginners")],
            default=1))

    # ---- finished size ------------------------------------------------
    if not a.stitches and not a.width_stitches and not a.width_inches:
        mode = _ask_choice(
            "how do you want to set the finished size?",
            [("inches", "by finished width in inches (height follows the image)"),
             ("stitches", "by stitch width (height follows the image)"),
             ("both", "by exact stitch grid WxH (may distort or crop)")],
            default=1)

        def pos_num(s: str) -> str | None:
            try:
                return None if float(s) > 0 else "must be greater than zero"
            except ValueError:
                return "enter a number"

        if mode == "inches":
            w_in = float(_ask("\nfinished width in inches", "12",
                              validate=pos_num))
            h_in = w_in * aspect
            print(f"  -> {w_in:.1f}\" x {h_in:.1f}\" "
                  f"({w_in / 12:.2f}ft x {h_in / 12:.2f}ft) at {a.count}-count")
            print(f"  -> {round(w_in * a.count)} x {round(w_in * a.count * aspect)} "
                  f"stitches, {round(w_in * a.count) * round(w_in * a.count * aspect):,} total")
            a.width_inches = w_in
        elif mode == "stitches":
            sw_ = int(_ask("\nstitch width", "200",
                           validate=lambda s: None if s.isdigit() and int(s) > 0
                           else "enter a whole number above zero"))
            print(f"  -> {sw_} x {round(sw_ * aspect)} stitches, "
                  f"{sw_ / a.count:.1f}\" x {sw_ * aspect / a.count:.1f}\" "
                  f"at {a.count}-count")
            a.width_stitches = sw_
        else:
            def grid_ok(s: str) -> str | None:
                try:
                    w, h = parse_stitches(s)
                except (argparse.ArgumentTypeError, ValueError):
                    return "use WIDTHxHEIGHT, e.g. 400x314"
                return None if w > 0 and h > 0 else "both must be above zero"

            got = _ask("\nstitch grid WxH", f"200x{round(200 * aspect)}",
                       validate=grid_ok)
            a.stitches = parse_stitches(got)
            gw, gh = a.stitches
            want = gh / gw
            if abs(want - aspect) > 0.02:
                print(f"  note: that grid is {want:.3f} but the image is "
                      f"{aspect:.3f}, so the picture will be stretched to fit")
            print(f"  -> {gw / a.count:.1f}\" x {gh / a.count:.1f}\" "
                  f"at {a.count}-count, {gw * gh:,} stitches")

    # ---- palette ------------------------------------------------------
    if not reused and not a.palette:
        keys = sorted(Path().glob("palette*.csv"))
        opts = [("auto", "let chartify choose the colours (original design)")]
        opts += [(str(k), f"reuse the existing key {k.name}") for k in keys[:3]]
        choice = _ask_choice("symbols and colours", opts, default=1)
        if choice != "auto":
            a.palette = Path(choice)

    if not reused and not a.palette and a.colors == 60:
        choice = _ask_choice(
            "how many thread colours?",
            [("60", "60 -- smooth shading, good for painterly images"),
             ("40", "40 -- simpler, flatter, fewer threads to buy"),
             ("90", "90 -- very smooth, but many near-duplicate threads"),
             ("25", "25 -- bold and posterised"),
             ("custom", "pick your own number")],
            default=1)
        if choice == "custom":
            a.colors = int(_ask(
                f"how many thread colours (max {len(SYMBOLS)})", "60",
                validate=lambda s: None if s.isdigit() and 0 < int(s) <= len(SYMBOLS)
                else f"enter a whole number from 1 to {len(SYMBOLS)}"))
        else:
            a.colors = int(choice)

    # ---- pages --------------------------------------------------------
    if not reused and a.squares_per_page == "5":
        a.squares_per_page = _ask_choice(
            "page size (big squares of 10 stitches per sheet)",
            [("5", "5x5 squares = 50x50 stitches per sheet"),
             ("3", "3x3 squares = 30x30 -- less daunting, more sheets"),
             ("6x8", "6x8 squares = 60x80 -- matches many printed kits"),
             ("8", "8x8 squares = 80x80 -- fewer, denser sheets")],
            default=1)

    # ---- binder margin -------------------------------------------------
    if not reused and a.margin_inches == 0.5:
        a.margin_inches = float(_ask_choice(
            "left margin for hole-punching into a binder",
            [("0.5", "0.5\" -- fits a standard 3-hole punch, least blank space"),
             ("0.75", "0.75\" -- standard binder margin, a bit more blank space"),
             ("1", "1\" -- extra room, more blank space below the chart"),
             ("0", "0 -- no margin, chart fills the page")],
            default=1))

    # ---- output -------------------------------------------------------
    if a.outdir == "chart":
        a.outdir = _ask("\noutput folder", _suggest_outdir(src))
    out = Path(a.outdir).expanduser()
    if out.exists() and any(out.iterdir()):
        if _ask_choice(
                f"'{out}' already exists and is not empty",
                [("keep", "write into it anyway (existing files may be overwritten)"),
                 ("new", "pick a different folder")],
                default=1) == "new":
            a.outdir = _ask("new output folder", _suggest_outdir(src))

    if not reused and a.format == "pdf":
        a.format = _ask_choice(
            "output format",
            [("pdf", "single PDF booklet -- cover, key, map, all pages in order"),
             ("png", "PNG -- one crisp image file per sheet"),
             ("jpg", "JPEG -- one smaller/softer image file per sheet")],
            default=1)

    # Name the sheets after the picture, not the generic default, so files from
    # different runs stay tellable apart once they're printed.
    if a.prefix == "chart":
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "-"
                       for ch in src.stem).strip("-").lower()
        a.prefix = _ask("sheet filename prefix", stem or "chart")

    settings_existed = _settings_path(src).exists()
    if not settings_existed or _ask_choice(
            "save these settings for next time?",
            [("yes", "save/overwrite the settings file for this image"),
             ("no", "leave the existing settings file as is")],
            default=1) == "yes":
        _save_settings(src, a)
    return a


def parse_squares(text: str) -> tuple[int, int]:
    """Parse --squares-per-page: either 'N' (square) or 'WxH'."""
    t = text.lower().strip()
    try:
        if "x" in t:
            w, _, h = t.partition("x")
            across, down = int(w), int(h)
        else:
            across = down = int(t)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--squares-per-page: expected N or WxH, got {text!r}")
    if across < 1 or down < 1:
        raise argparse.ArgumentTypeError(
            "--squares-per-page must be at least 1 square in each direction")
    return across, down


def main(argv: list[str] | None = None) -> int:
    global PAGE_W, PAGE_H

    p = argparse.ArgumentParser(
        description="Convert an image into a printable cross-stitch symbol chart.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1])
    p.add_argument("input", nargs="?",
                   help="source image; omit it to be prompted for everything")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="ask for any option not given on the command line "
                        "(implied when INPUT is omitted)")
    p.add_argument("-o", "--outdir", default="chart", help="output directory (default: chart)")

    size = p.add_argument_group("finished size (pick one)")
    size.add_argument("--stitches", type=parse_stitches, metavar="WxH",
                      help="explicit stitch count, e.g. 400x314")
    size.add_argument("--width-stitches", type=int, help="stitch width; height follows aspect")
    size.add_argument("--width-inches", type=float, help="finished width in inches")
    size.add_argument("--count", type=int, default=14,
                      help="fabric count (stitches per inch) for --width-inches (default: 14)")

    p.add_argument("-c", "--colors", type=int, default=60,
                   help=f"maximum thread colours (default: 60, max "
                        f"{len(SYMBOLS)} -- one symbol per colour); "
                        f"ignored with --palette")
    p.add_argument("--palette", type=Path, metavar="CSV",
                   help="use a fixed symbol/dmc palette CSV instead of choosing "
                        "and lettering colours automatically -- this is how you "
                        "match an existing chart's symbols")
    p.add_argument("--squares-per-page", default="5", metavar="N|WxH",
                   help="big squares per page: one number for a square page, "
                        "or WxH for different across/down counts "
                        "(default: 5, i.e. a 5x5 grid of 10-stitch squares; "
                        "'3' gives 3x3, '6x8' gives 6 across by 8 down)")
    p.add_argument("--block", type=int, default=10,
                   help="stitches per big square (default: 10)")
    p.add_argument("--cell", type=int, default=26,
                   help="pixels per stitch cell in the output (default: 26)")
    p.add_argument("--prefix", default="chart", help="tile filename prefix (default: chart)")
    p.add_argument("--format", default="pdf", choices=("pdf", "jpg", "png"),
                   help="output format (default: pdf, written as a single "
                        "multi-page booklet -- cover, key, map, then every "
                        "chart page in order. Bakes the physical page size "
                        "into the file itself, which sidesteps printers/"
                        "drivers that mis-scale a raster image's DPI. "
                        "'jpg'/'png' write one image file per sheet instead")
    p.add_argument("--no-context", action="store_true",
                   help="omit the faded edge strips showing the adjoining "
                        "pages' stitches and the page locator map")
    p.add_argument("--margin-inches", type=float, default=0.5,
                   help="blank strip added down the left edge of every "
                        "printed page, for hole-punching into a binder "
                        "(default: 0.5; common choices are 0.5, 0.75, 1; "
                        "use 0 for no margin). Chart pages are wider than "
                        "tall, so a bigger margin means more blank space "
                        "below the chart to keep the page portrait-shaped")
    p.add_argument("--dpi", type=int, default=300,
                   help="resolution assumed when converting --margin-inches "
                        "to pixels, and when sizing pages to --page-size "
                        "(default: 300)")
    p.add_argument("--page-size", default="letter",
                   choices=("letter", "a4", "none"),
                   help="physical sheet every page is centred on, so "
                        "printing at actual size (no scaling) fills the "
                        "sheet instead of leaving the chart small in a "
                        "corner (default: letter; 'none' skips this and "
                        "writes each page at its natural content size)")

    p.add_argument("--learn-from", type=Path, metavar="CSV",
                   help="CSV of col,row,symbol cells from a real chart; the "
                        "script learns this designer's colour->thread mapping "
                        "from them instead of using generic DMC matching. "
                        "Requires --palette")

    edge = p.add_argument_group("source edges")
    edge.add_argument("--crop", metavar="L,T,R,B",
                      help="trim this many source pixels off each edge before "
                           "charting, e.g. 0,1,0,0 to drop one white top row")
    edge.add_argument("--no-autocrop", action="store_true",
                      help="keep near-white scan edges instead of trimming them")

    args = p.parse_args(argv)

    # Prompting needs a terminal on both ends; piped input would hang forever.
    if args.interactive or args.input is None:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            p.error("INPUT is required when not running interactively")
        args = interactive(args)

    src = Path(args.input).expanduser()
    if not src.exists():
        sys.exit(f"no such file: {src}")

    if args.learn_from and not args.palette:
        sys.exit("--learn-from needs --palette: the training cells are symbols, "
                 "so the symbol->thread key has to be fixed")
    if args.learn_from and not args.learn_from.exists():
        sys.exit(f"no such file: {args.learn_from}")

    if not args.palette and args.colors > len(SYMBOLS):
        sys.exit(f"--colors {args.colors} needs one symbol per colour, but "
                 f"only {len(SYMBOLS)} are defined; lower --colors "
                 f"to {len(SYMBOLS)} or below")

    img = Image.open(src).convert("RGB")
    raw_w, raw_h = img.size

    # Trim source edges before anything else, so the stitch grid is laid over
    # actual artwork rather than a scan border.
    if args.crop:
        try:
            l, t, r, b = (int(v) for v in args.crop.split(","))
        except ValueError:
            sys.exit("--crop wants four integers: L,T,R,B")
    elif args.no_autocrop:
        l = t = r = b = 0
    else:
        l, t, r, b = detect_border(img)
        if any((l, t, r, b)):
            print(f"autocrop    : trimmed near-white edges "
                  f"L{l} T{t} R{r} B{b} (--no-autocrop to keep)")

    if any((l, t, r, b)):
        img = img.crop((l, t, raw_w - r, raw_h - b))

    ow, oh = img.size
    if (ow, oh) != (raw_w, raw_h):
        print(f"source crop : {raw_w}x{raw_h} -> {ow}x{oh}px")

    # Work out the stitch grid.
    if args.stitches:
        sw, sh = args.stitches
    elif args.width_stitches:
        sw = args.width_stitches
        sh = round(sw * oh / ow)
    elif args.width_inches:
        sw = round(args.width_inches * args.count)
        sh = round(sw * oh / ow)
    else:
        sw = 200
        sh = round(sw * oh / ow)

    try:
        sq_across, sq_down = parse_squares(args.squares_per_page)
    except argparse.ArgumentTypeError as exc:
        sys.exit(str(exc))
    PAGE_W = sq_across * args.block
    PAGE_H = sq_down * args.block

    print(f"source      : {src.name} ({ow}x{oh}px)")
    print(f"stitch grid : {sw}x{sh} = {sw * sh:,} stitches")
    print(f"fabric      : {args.count}-count -> "
          f"{sw / args.count:.1f}\" x {sh / args.count:.1f}\" finished")

    # Resample to the stitch grid. BOX averages the source area per stitch,
    # which reads better than bicubic here -- no ringing, no invented colours.
    small = img.resize((sw, sh), Image.BOX)
    arr = np.asarray(small, dtype=np.float64).reshape(-1, 3)

    # Deduplicate before the expensive colour maths.
    uniq, inverse = np.unique(arr, axis=0, return_inverse=True)
    print(f"palette     : matching {len(uniq):,} distinct colours to DMC...")
    uniq_lab = srgb_to_lab(uniq)

    # Only a --learn-from run distinguishes read cells from predicted ones;
    # otherwise every cell is equally derived from the image.
    confirmed = None

    if args.palette:
        # Fixed palette: every stitch maps to the nearest thread in the given
        # list, and each thread keeps the symbol the palette file assigns it.
        # Threads are NOT dropped or re-lettered -- the notation has to stay
        # stable to match the chart it came from.
        palette = load_palette_csv(args.palette)
        print(f"palette file: {args.palette.name} "
              f"({len(palette)} threads, fixed symbols)")
        if args.learn_from:
            matrix = learn_from_cells(args.learn_from,
                                      np.asarray(small, dtype=np.float64),
                                      palette)
            confirmed = confirmed_mask(args.learn_from, sh, sw)
            print(f"read cells  : {int(confirmed.sum())} of {sh * sw} "
                  f"({100.0 * confirmed.sum() / (sh * sw):.1f}%) "
                  f"-- the rest are predicted")
        else:
            uniq_idx = map_to_palette(uniq_lab, palette)
            matrix = uniq_idx[inverse].reshape(sh, sw)
        for n, thread in enumerate(palette):
            thread.stitches = int((matrix == n).sum())
        unused = [t for t in palette if t.stitches == 0]
        if unused:
            print(f"  note: {len(unused)} palette threads unused in this image "
                  f"({', '.join(t.code for t in unused[:8])}"
                  f"{'...' if len(unused) > 8 else ''})")
    else:
        palette, _, _ = choose_palette(uniq_lab, args.colors)
        uniq_idx = map_to_palette(uniq_lab, palette)
        matrix = uniq_idx[inverse].reshape(sh, sw)

        # Drop any thread that survived selection but ended up unused, then
        # assign symbols in descending area order so the commonest colours get
        # the clearest glyphs.
        counts = np.bincount(matrix.ravel(), minlength=len(palette))
        keep = [i for i in np.argsort(counts)[::-1] if counts[i] > 0]
        remap = {old: new for new, old in enumerate(keep)}
        matrix = np.vectorize(remap.__getitem__)(matrix).astype(np.int32)
        palette = [palette[i] for i in keep]

        if len(palette) > len(SYMBOLS):
            sys.exit(f"need {len(palette)} symbols but only {len(SYMBOLS)} "
                     f"defined; lower --colors")

        for n, thread in enumerate(palette):
            thread.symbol = SYMBOLS[n]
            thread.stitches = int((matrix == n).sum())

    # Skein estimates depend on the fabric count, so tell every thread which
    # count this run is for before anything reads `.skeins`.
    for thread in palette:
        thread.count = args.count

    print(f"threads     : {len(palette)} DMC colours")
    total_skeins = sum(t.skeins for t in palette if t.stitches)
    print(f"floss       : ~{total_skeins} skeins total "
          f"(2 strands on {args.count}-count)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Tile the chart. Origin is top-left; x increases right, y increases down.
    cols_of_pages = math.ceil(sw / PAGE_W)
    rows_of_pages = math.ceil(sh / PAGE_H)
    print(f"pages       : {cols_of_pages} across x {rows_of_pages} down "
          f"({sq_across}x{sq_down} squares = {PAGE_W}x{PAGE_H} stitches each)")

    # A final band only a few stitches wide is a whole sheet of paper carrying
    # almost nothing; nudging the stitch count usually removes it entirely.
    tail_w = sw - (cols_of_pages - 1) * PAGE_W
    tail_h = sh - (rows_of_pages - 1) * PAGE_H
    for who, tail, page, total, npages in (
            ("column", tail_w, PAGE_W, sw, cols_of_pages),
            ("row", tail_h, PAGE_H, sh, rows_of_pages)):
        if npages > 1 and tail <= page // 5:
            print(f"  note: last page {who} is only {tail} stitches wide "
                  f"-- trimming {total} to {total - tail} would drop that "
                  f"nearly-empty band of sheets")

    ext = args.format
    written = []
    # PDF pages are spooled to disk as they are finished rather than held in
    # a list. A Letter sheet at 300dpi is 2550x3300 RGB -- about 25MB -- so a
    # 40-page booklet would need ~1GB resident before the first byte was
    # written, which overruns the heap in a memory-capped environment (the
    # browser/WASM build hits MemoryError inside the LANCZOS resize). Spooling
    # keeps peak usage at roughly one page, and Pillow reopens the spooled
    # files lazily at save time.
    booklet: list[Path] = []
    spool = outdir / ".pages"
    save_opts = {"quality": 95} if ext == "jpg" else {}
    save_opts["dpi"] = (args.dpi, args.dpi)

    page_dims = {"letter": (8.5, 11.0), "a4": (8.27, 11.69)}.get(args.page_size)

    def save(art: Image.Image, name: str) -> Path | None:
        art = add_binder_margin(art, inches=args.margin_inches, dpi=args.dpi)
        if page_dims:
            art = fit_to_page(art, page_w_in=page_dims[0], page_h_in=page_dims[1],
                              margin_in=args.margin_inches, dpi=args.dpi)
        # A single multi-page PDF is the booklet; per-sheet files would just
        # be one-page duplicates of pages already collated in it.
        if ext == "pdf":
            spool.mkdir(parents=True, exist_ok=True)
            # PNG keeps the page pixel-exact, so spooling cannot alter output.
            page_path = spool / f"{len(booklet):04d}.png"
            art.save(page_path, "PNG", compress_level=1)
            booklet.append(page_path)
            art.close()
            return None
        path = outdir / name
        art.save(path, **save_opts)
        return path

    # Front matter is written first and numbered 00a-00e so a plain
    # filename sort collates the booklet: cover, finished preview,
    # DMC-order shopping list, key, map, then page01 onward. Writing them
    # in that order also means a combined PDF comes out in reading order
    # without a separate resort step. The preview follows the cover
    # (which carries the usage-order thread list) because it's the only
    # sheet showing actual colours -- the natural next thing to check
    # after "what threads do I need"; the DMC-order list follows that, as
    # the version to actually take to the store.
    written.append(save(render_cover(
        matrix, palette, source=src, count=args.count,
        page_w=PAGE_W, page_h=PAGE_H, block=args.block,
        prefix=f"{args.prefix}-", palette_file=args.palette,
        learn_file=args.learn_from, confirmed=confirmed),
        f"{args.prefix}-00a-cover.{ext}"))

    written.append(save(render_preview_page(matrix, palette, source=src),
                        f"{args.prefix}-00b-preview.{ext}"))
    if ext != "pdf":
        render_preview(matrix, palette).save(
            outdir / f"{args.prefix}-preview-flat.{ext}", **save_opts)

    written.append(save(render_shopping_list(palette),
                        f"{args.prefix}-00c-shopping-list.{ext}"))

    written.append(save(render_key(palette), f"{args.prefix}-00d-key.{ext}"))

    written.append(save(render_page_map(
        matrix, palette, page_w=PAGE_W, page_h=PAGE_H,
        prefix=f"{args.prefix}-"), f"{args.prefix}-00e-page-map.{ext}"))

    # Sheets are numbered in reading order so they collate into a booklet, and
    # the grid position stays in the name because that is what tells you where
    # the sheet belongs in the design.
    total_pages = cols_of_pages * rows_of_pages
    digits = max(2, len(str(total_pages)))
    for py in range(rows_of_pages):
        for px in range(cols_of_pages):
            page_no = py * cols_of_pages + px + 1
            stem = (f"{args.prefix}-page{page_no:0{digits}d}"
                    f"-x{px + 1}y{py + 1}")
            tile = render_tile(matrix, palette,
                               x0=px * PAGE_W, y0=py * PAGE_H,
                               cell=args.cell, block=args.block,
                               label=f"Page {page_no} of {total_pages}"
                                     f"   (x{px + 1}y{py + 1})",
                               confirmed=confirmed,
                               neighbours=(px + 1, py + 1,
                                           cols_of_pages, rows_of_pages)
                               if not args.no_context else None,
                               prefix=f"{args.prefix}-")
            written.append(save(tile, f"{stem}.{ext}"))

    if ext == "pdf":
        pdf_path = outdir / f"{args.prefix}.pdf"
        # Reopen the spooled pages lazily. Pillow reads each appended image
        # as it writes it, so only the first page plus the one being encoded
        # are resident -- the whole point of spooling above.
        # Pillow's writer holds every page decoded and resident -- about
        # 25MB per Letter sheet at 300dpi -- so it needs over a gigabyte for
        # a large booklet. Past a modest page count the streaming writer is
        # used instead: it produces the same pages (both JPEG-encode the
        # sheets) while keeping peak memory to a single page.
        PILLOW_PAGE_LIMIT = 12
        try:
            if len(booklet) > PILLOW_PAGE_LIMIT:
                write_pdf_streaming(booklet, pdf_path, dpi=args.dpi)
            else:
                first = Image.open(booklet[0])
                rest = [Image.open(p) for p in booklet[1:]]
                try:
                    first.save(pdf_path, save_all=True, append_images=rest,
                               resolution=args.dpi)
                finally:
                    first.close()
                    for im in rest:
                        im.close()
        except MemoryError:
            # Small booklet, but still no headroom -- fall back regardless.
            print("  note: not enough memory to collate the booklet at once; "
                  "writing it a page at a time")
            write_pdf_streaming(booklet, pdf_path, dpi=args.dpi)
        finally:
            # The spool is an implementation detail; don't leave it behind.
            for p in booklet:
                p.unlink(missing_ok=True)
            spool.rmdir()
        written = [pdf_path]

    # Machine-readable symbol matrix.
    with open(outdir / f"{args.prefix}-symbols.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row"] + [f"c{i + 1}" for i in range(sw)])
        for r in range(sh):
            w.writerow([r + 1] + [palette[int(i)].symbol for i in matrix[r]])

    with open(outdir / f"{args.prefix}-threads.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "dmc", "name", "stitches", "est_skeins", "hex"])
        for t in palette:
            w.writerow([t.symbol, t.code, t.name, t.stitches, t.skeins,
                        "#%02X%02X%02X" % t.rgb])

    if ext == "pdf":
        print(f"\nwrote {total_pages + 5}-page booklet (cover, preview, "
              f"shopping list, key, map, {total_pages} chart pages) "
              f"+ CSVs to {written[0]}")
    else:
        print(f"\nwrote {len(written)} chart pages + key, preview and CSVs to {outdir}/")
        print(f"  tiles run {args.prefix}-x1y1 .. {args.prefix}-x{cols_of_pages}y{rows_of_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
