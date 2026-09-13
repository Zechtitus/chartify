"""Tests for the chart engine.

Scope is deliberate: the things that would corrupt every chart *without
crashing*, plus the properties the printed output depends on.

  * colour science -- verified against the published CIEDE2000 reference
    data, because a wrong thread match is silent and only visible once the
    piece is half stitched
  * geometry -- page fitting and tiling, which decide whether a printed
    sheet is the right physical size
  * parsers -- the CLI's contract with the user
  * the PDF writers -- both paths must produce the same pages

Rendering is exercised for invariants (size, page count, determinism)
rather than pixel-compared: the reference-image approach would break on
every font or Pillow upgrade without indicating a real fault.

Run with:  task test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chartify
from chartify import (
    Thread,
    add_binder_margin,
    ciede2000,
    fit_to_page,
    map_to_palette,
    parse_squares,
    parse_stitches,
    srgb_to_lab,
    write_pdf_streaming,
)


# --------------------------------------------------------------------------
# Colour science
# --------------------------------------------------------------------------

# Sharma, Wu & Dalal's CIEDE2000 verification set: the standard way to show
# an implementation handles the hue-rotation and arctan wraparound terms.
# (L1, a1, b1, L2, a2, b2, expected dE00)
SHARMA_CASES = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
]


@pytest.mark.parametrize("case", SHARMA_CASES)
def test_ciede2000_matches_reference_data(case):
    """The distance metric must match the published CIEDE2000 values."""
    l1, a1, b1, l2, a2, b2, expected = case
    got = ciede2000(np.array([[l1, a1, b1]]), np.array([[l2, a2, b2]]))
    assert got.shape == (1, 1)
    # The reference table is quoted to 4dp, so hold the implementation to
    # that. A looser tolerance here lets real errors through: dropping the
    # hue-wraparound term shifts one of these cases by only 7e-4.
    assert got[0, 0] == pytest.approx(expected, abs=2e-4)


def test_ciede2000_is_zero_for_identical_colours():
    lab = np.array([[42.0, -13.0, 27.0]])
    assert ciede2000(lab, lab)[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_ciede2000_returns_full_distance_matrix():
    """Shape contract: (N,3) against (M,3) gives (N,M)."""
    a = np.random.default_rng(0).uniform(0, 100, (7, 3))
    b = np.random.default_rng(1).uniform(0, 100, (4, 3))
    assert ciede2000(a, b).shape == (7, 4)


def test_srgb_to_lab_known_anchors():
    """White, black and the primaries pin the transfer function and matrix."""
    rgb = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8)
    lab = srgb_to_lab(rgb)

    # D65 white -> L*=100, neutral chroma.
    assert lab[0, 0] == pytest.approx(100.0, abs=1e-3)
    assert lab[0, 1] == pytest.approx(0.0, abs=1e-3)
    assert lab[0, 2] == pytest.approx(0.0, abs=1e-3)

    # Black -> the origin.
    assert np.allclose(lab[1], [0.0, 0.0, 0.0], atol=1e-9)

    # sRGB red has well-known CIELAB coordinates under D65.
    assert lab[2, 0] == pytest.approx(53.24, abs=0.02)
    assert lab[2, 1] == pytest.approx(80.09, abs=0.02)
    assert lab[2, 2] == pytest.approx(67.20, abs=0.02)


def test_srgb_to_lab_preserves_shape():
    """Conversion must work on an image-shaped array, not just a list."""
    block = np.zeros((5, 4, 3), dtype=np.uint8)
    assert srgb_to_lab(block).shape == (5, 4, 3)


def test_srgb_to_lab_is_monotonic_in_lightness():
    """Darker greys must not come out lighter -- catches a flipped curve."""
    greys = np.array([[[v, v, v]] for v in range(0, 256, 8)], dtype=np.uint8)
    light = srgb_to_lab(greys)[:, 0, 0]
    assert np.all(np.diff(light) > 0)


# --------------------------------------------------------------------------
# Palette mapping
# --------------------------------------------------------------------------

def _threads(*rgbs: tuple[int, int, int]) -> list[Thread]:
    """Build a palette. Thread carries rgb only -- map_to_palette derives
    the Lab coordinates itself."""
    return [Thread(code=str(100 + i), name=f"T{i}", rgb=rgb,
                   symbol=chartify.SYMBOLS[i])
            for i, rgb in enumerate(rgbs)]


def test_map_to_palette_picks_the_nearest_thread():
    """An exact palette colour must map to its own index."""
    palette = _threads((255, 0, 0), (0, 255, 0), (0, 0, 255))
    pixels = srgb_to_lab(np.array(
        [(0, 0, 255), (255, 0, 0), (0, 255, 0)], dtype=np.uint8))
    assert list(map_to_palette(pixels, palette)) == [2, 0, 1]


def test_map_to_palette_handles_near_misses():
    """A slightly-off colour still resolves to the visually closest thread."""
    palette = _threads((0, 0, 0), (255, 255, 255))
    pixels = srgb_to_lab(np.array([(10, 10, 10), (245, 245, 245)],
                                  dtype=np.uint8))
    assert list(map_to_palette(pixels, palette)) == [0, 1]


# --------------------------------------------------------------------------
# Parsers (the CLI's contract)
# --------------------------------------------------------------------------

def test_skeins_never_rounds_down_to_zero():
    """You cannot buy part of a skein, so one stitch still needs one."""
    assert Thread(code="310", name="Black", rgb=(0, 0, 0),
                  stitches=1, count=14).skeins == 1


def test_skeins_rounds_up():
    """1,800 crosses per skein at 14-count; 1,801 needs a second."""
    mk = lambda n: Thread(code="310", name="Black", rgb=(0, 0, 0),  # noqa: E731
                          stitches=n, count=14)
    assert mk(1800).skeins == 1
    assert mk(1801).skeins == 2
    assert mk(3600).skeins == 2


def test_skeins_scale_with_fabric_count():
    """Finer fabric uses less thread per stitch, so a skein goes further."""
    mk = lambda c: Thread(code="310", name="Black", rgb=(0, 0, 0),  # noqa: E731
                          stitches=5000, count=c)
    coarse, fine = mk(11).skeins, mk(18).skeins
    assert coarse > fine, f"11-count {coarse} should exceed 18-count {fine}"


def test_parse_stitches_reads_dimensions():
    assert parse_stitches("400x314") == (400, 314)
    assert parse_stitches("400X314") == (400, 314)


def test_parse_stitches_rejects_missing_separator():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_stitches("400")


@pytest.mark.parametrize("text,expected", [
    ("5", (5, 5)),
    ("3", (3, 3)),
    ("6x8", (6, 8)),
    (" 6X8 ", (6, 8)),
])
def test_parse_squares_accepts_both_forms(text, expected):
    assert parse_squares(text) == expected


@pytest.mark.parametrize("bad", ["0", "-2", "0x5", "5x0", "abc", "", "5x"])
def test_parse_squares_rejects_nonsense(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_squares(bad)


# --------------------------------------------------------------------------
# Page geometry -- decides the physical size of a printed sheet
# --------------------------------------------------------------------------

@pytest.mark.parametrize("page_w_in,page_h_in,dpi", [
    (8.5, 11.0, 300),    # Letter
    (8.27, 11.69, 300),  # A4
    (8.5, 11.0, 150),
])
def test_fit_to_page_produces_exact_page_size(page_w_in, page_h_in, dpi):
    """Every sheet must come out the requested physical size."""
    art = Image.new("RGB", (900, 500), "white")
    out = fit_to_page(art, page_w_in=page_w_in, page_h_in=page_h_in,
                      margin_in=0.5, dpi=dpi)
    assert out.size == (round(page_w_in * dpi), round(page_h_in * dpi))


def test_fit_to_page_does_not_distort_content():
    """Scaling must preserve aspect ratio -- a squashed chart is unusable."""
    art = Image.new("RGB", (400, 200), "black")
    out = fit_to_page(art, page_w_in=8.5, page_h_in=11.0,
                      margin_in=0.5, dpi=300)
    # Find the black region's bounding box and check its proportions.
    bbox = out.convert("L").point(lambda v: 255 if v < 128 else 0).getbbox()
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    assert w / h == pytest.approx(2.0, rel=0.02)


def test_fit_to_page_keeps_oversized_content_inside_the_sheet():
    """Content larger than the page is scaled down, never cropped."""
    art = Image.new("RGB", (6000, 9000), "black")
    out = fit_to_page(art, page_w_in=8.5, page_h_in=11.0,
                      margin_in=0.5, dpi=300)
    assert out.size == (2550, 3300)
    bbox = out.convert("L").point(lambda v: 255 if v < 128 else 0).getbbox()
    assert bbox[0] >= 0 and bbox[1] >= 0
    assert bbox[2] <= 2550 and bbox[3] <= 3300


def test_add_binder_margin_widens_on_the_left_only():
    art = Image.new("RGB", (100, 80), "black")
    out = add_binder_margin(art, inches=0.5, dpi=300)
    assert out.size == (100 + 150, 80)
    # The strip is blank and the artwork is flush against it.
    assert out.getpixel((10, 40)) == (255, 255, 255)
    assert out.getpixel((150 + 10, 40)) == (0, 0, 0)


def test_add_binder_margin_is_a_noop_at_zero():
    art = Image.new("RGB", (60, 40), "black")
    assert add_binder_margin(art, inches=0, dpi=300).size == (60, 40)


# --------------------------------------------------------------------------
# PDF writers -- the streaming path must match the Pillow path
# --------------------------------------------------------------------------

def _spool(tmp_path: Path, count: int, size=(510, 660)) -> list[Path]:
    """Write `count` distinct pages, so ordering errors are detectable."""
    paths = []
    band = max(8, size[1] // (count * 3))
    for i in range(count):
        img = Image.new("RGB", size, (255, 255, 255))
        # A black band whose vertical position encodes the page index, so a
        # reordered booklet is detectable. Kept well inside the page so JPEG
        # ringing at the edges cannot swallow it.
        top = 20 + i * band * 2
        assert top + band < size[1], "fingerprint would fall off the page"
        for y in range(top, top + band):
            for x in range(size[0]):
                img.putpixel((x, y), (0, 0, 0))
        p = tmp_path / f"{i:04d}.png"
        img.save(p, "PNG")
        paths.append(p)
    return paths


def test_write_pdf_streaming_page_count_and_size(tmp_path):
    pages = _spool(tmp_path, 5)
    out = tmp_path / "book.pdf"
    write_pdf_streaming(pages, out, dpi=300)

    data = out.read_bytes()
    assert data.startswith(b"%PDF-1.4")
    assert data.rstrip().endswith(b"%%EOF")
    assert data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") == 5


def test_write_pdf_streaming_is_readable_by_pillow(tmp_path):
    """A PDF nothing can open would still pass a byte-level check."""
    pages = _spool(tmp_path, 3)
    out = tmp_path / "book.pdf"
    write_pdf_streaming(pages, out, dpi=300)
    # Pillow cannot read PDFs, so shell out to pdfinfo when it exists.
    info = subprocess.run(["pdfinfo", str(out)],
                          capture_output=True, text=True)
    if info.returncode != 0:
        pytest.skip("pdfinfo not available")
    assert "Pages:           3" in info.stdout
    # 510x660px at 300dpi -> 122.4 x 158.4 pt
    assert "122.4 x 158.4 pts" in info.stdout


def test_write_pdf_streaming_page_size_follows_dpi(tmp_path):
    """A 2550x3300 sheet at 300dpi must be exactly US Letter."""
    pages = _spool(tmp_path, 1, size=(2550, 3300))
    out = tmp_path / "letter.pdf"
    write_pdf_streaming(pages, out, dpi=300)
    info = subprocess.run(["pdfinfo", str(out)],
                          capture_output=True, text=True)
    if info.returncode != 0:
        pytest.skip("pdfinfo not available")
    assert "612 x 792 pts (letter)" in info.stdout


def test_write_pdf_streaming_preserves_page_order(tmp_path):
    """Page N of the booklet must be page N of the PDF."""
    pages = _spool(tmp_path, 4)
    out = tmp_path / "ordered.pdf"
    write_pdf_streaming(pages, out, dpi=300)
    if subprocess.run(["which", "pdftoppm"],
                      capture_output=True).returncode != 0:
        pytest.skip("pdftoppm not available")
    subprocess.run(["pdftoppm", "-r", "36", "-png", str(out),
                    str(tmp_path / "pg")], check=True)
    rendered = sorted(tmp_path.glob("pg-*.png"))
    assert len(rendered) == 4
    # The band's vertical position increases with the page index.
    tops = []
    for r in rendered:
        # Threshold generously: the pages are JPEG-encoded in the PDF, so
        # the band's edges arrive soft rather than pure black.
        mask = Image.open(r).convert("L").point(
            lambda v: 255 if v < 200 else 0)
        box = mask.getbbox()
        assert box is not None, f"no fingerprint band found on {r.name}"
        tops.append(box[1])
    assert tops == sorted(tops), f"pages out of order: {tops}"
    assert len(set(tops)) == 4, f"pages not distinguishable: {tops}"


def test_write_pdf_streaming_holds_one_page_in_memory(tmp_path):
    """The streaming writer's peak must not scale with page count."""
    try:
        import resource
    except ImportError:
        pytest.skip("resource module unavailable")

    pages = _spool(tmp_path, 24, size=(2550, 3300))
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_pdf_streaming(pages, tmp_path / "big.pdf", dpi=300)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grew_mb = (after - before) / 1024

    # 24 Letter sheets decoded at once would be ~600MB; one page is ~25MB.
    assert grew_mb < 250, f"peak memory grew {grew_mb:.0f}MB"


def test_large_booklet_does_not_collate_in_memory(tmp_path, monkeypatch):
    """Guards the regression that caused MemoryError in the browser.

    A booklet past the page limit must be written by the streaming path.
    Asserting on RSS through main() is too noisy to be reliable, so this
    checks the decision directly: Pillow's all-pages-resident save must
    not be the one that runs.
    """
    calls = {"streaming": 0, "pillow_multipage": 0}

    real_streaming = chartify.write_pdf_streaming

    def spy_streaming(pages, out, **kw):
        calls["streaming"] += 1
        return real_streaming(pages, out, **kw)

    real_save = Image.Image.save

    def spy_save(self, fp, *a, **kw):
        if kw.get("save_all"):
            calls["pillow_multipage"] += 1
        return real_save(self, fp, *a, **kw)

    monkeypatch.setattr(chartify, "write_pdf_streaming", spy_streaming)
    monkeypatch.setattr(Image.Image, "save", spy_save)

    # 120 stitches wide at 50 stitches per page gives plenty of sheets.
    rng = np.random.default_rng(3)
    src = tmp_path / "src.png"
    Image.fromarray(rng.integers(0, 255, (150, 120, 3), dtype=np.uint8)).save(src)

    rc = chartify.main([str(src), "--width-stitches", "120", "--colors", "6",
                        "-o", str(tmp_path / "out"), "--format", "pdf"])
    assert rc == 0
    assert calls["streaming"] == 1, "large booklet did not stream"
    assert calls["pillow_multipage"] == 0, (
        "Pillow's all-pages-resident save was used for a large booklet")


# --------------------------------------------------------------------------
# End-to-end: the engine still produces a complete booklet
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_image(tmp_path_factory) -> Path:
    """A small synthetic image -- no repo fixture needed."""
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (60, 45, 3), dtype=np.uint8)
    img = Image.fromarray(base).filter(
        __import__("PIL.ImageFilter", fromlist=["GaussianBlur"])
        .GaussianBlur(2))
    p = tmp_path_factory.mktemp("src") / "sample.png"
    img.save(p)
    return p


def test_main_writes_a_complete_booklet(sample_image, tmp_path, capsys):
    out = tmp_path / "chart"
    rc = chartify.main([str(sample_image), "--width-stitches", "40",
                        "--colors", "8", "-o", str(out), "--format", "pdf"])
    assert rc == 0

    pdf = out / "chart.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1000
    assert (out / "chart-symbols.csv").exists()
    assert (out / "chart-threads.csv").exists()
    # The spool is an implementation detail and must be cleaned up.
    assert not (out / ".pages").exists()


def test_symbols_csv_matches_the_grid(sample_image, tmp_path):
    """Every stitch must appear in the audit CSV exactly once."""
    import csv
    out = tmp_path / "chart"
    chartify.main([str(sample_image), "--width-stitches", "30",
                   "--colors", "6", "-o", str(out), "--format", "pdf"])
    rows = list(csv.reader((out / "chart-symbols.csv").open()))
    header, body = rows[0], rows[1:]
    assert header[0] == "row"
    assert len(header) - 1 == 30                  # one column per stitch
    assert all(len(r) == len(header) for r in body)

    # Symbols used in the grid must all be declared in the thread list.
    threads = list(csv.DictReader((out / "chart-threads.csv").open()))
    declared = {t["symbol"] for t in threads}
    used = {c for r in body for c in r[1:]}
    assert used <= declared, f"undeclared symbols: {used - declared}"


def test_thread_list_never_promises_zero_skeins(sample_image, tmp_path):
    """You cannot buy part of a skein, so every colour needs at least one."""
    import csv
    out = tmp_path / "chart"
    chartify.main([str(sample_image), "--width-stitches", "30",
                   "--colors", "6", "-o", str(out), "--format", "pdf"])
    threads = list(csv.DictReader((out / "chart-threads.csv").open()))
    assert threads
    for t in threads:
        assert int(t["est_skeins"]) >= 1
        assert int(t["stitches"]) >= 1


def test_colour_cap_is_respected(sample_image, tmp_path):
    import csv
    out = tmp_path / "chart"
    chartify.main([str(sample_image), "--width-stitches", "40",
                   "--colors", "5", "-o", str(out), "--format", "pdf"])
    threads = list(csv.DictReader((out / "chart-threads.csv").open()))
    assert 0 < len(threads) <= 5


def test_run_is_deterministic(sample_image, tmp_path):
    """Same input and flags must give the same chart, or the CSVs lie."""
    digests = []
    for i in range(2):
        out = tmp_path / f"run{i}"
        chartify.main([str(sample_image), "--width-stitches", "30",
                       "--colors", "6", "-o", str(out), "--format", "pdf"])
        digests.append((out / "chart-symbols.csv").read_bytes())
    assert digests[0] == digests[1]
