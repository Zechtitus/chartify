# Chartify

Turn any picture into a cross-stitch chart you can actually print and stitch
from.

Feed it an image and it resamples the picture to a stitch grid, matches every
stitch to the nearest real DMC stranded-cotton thread, assigns a symbol to each
colour, and writes a collated PDF booklet: tiled chart pages, a colour key, a
shopping list, and a preview of the finished piece.

**Try it in your browser: <https://zechtitus.github.io/chartify/>** — no
install, and your picture never leaves your machine.

---

## What you get

One PDF, in reading order, ready to print:

| Sheet | What it's for |
|---|---|
| **Cover** | Thread list in the order you'll use them, plus the chart's vital statistics |
| **Preview** | What the finished piece should look like in colour |
| **Shopping list** | Threads by DMC number, with skein estimates — the sheet to take to the shop |
| **Key** | Symbol → DMC code → name, with a printed swatch of each colour |
| **Page map** | How the sheets fit together |
| **Chart pages** | The symbol grid, tiled across as many sheets as it takes |

Plus two CSVs: `-symbols.csv` (every stitch as a symbol, so the decode is
auditable) and `-threads.csv` (the thread list with stitch counts and skein
estimates).

Chart pages carry the first two stitches of each adjoining page down their
edges — drawn as real symbol cells but greyed and set off by a gap — so you can
match notation across a seam without unfolding the next sheet.

## Two ways to run it

Same engine (`chartify.py`) behind both.

```sh
task serve                                   # web app, http://localhost:8000
task chart                                   # CLI, interactive prompts
task chart -- "photo.jpg --colors 40"        # CLI, driven by flags
task --list                                  # everything else
```

The web app is the friendlier one; the CLI has the full option set and runs
about six times faster. Use the CLI for large charts.

## Design notes

**Colours are matched in CIELAB, not RGB.** RGB-Euclidean matching picks
visibly wrong threads — it goes badly astray in skin tones and muted greens,
which is exactly where a portrait lives or dies.

**Symbols are chosen for legibility at print size.** The 163-glyph alphabet is
ordered roughly by clarity — plain capitals first, then digits and lowercase,
then extended glyphs — and the commonest colour gets the clearest symbol.
Whole families of lookalikes (box-drawing pieces, filled/hollow circle sets,
dice faces) are deliberately excluded: they read as the same mark in a cell a
few millimetres across. Every glyph is verified present in the bundled font.

**Pages are sized in squares, not pixels.** `--squares-per-page` counts big
squares (10 stitches each by default), so a page is a whole number of squares
and the grid never splits awkwardly across a seam.

**Skein estimates are deliberately pessimistic.** They're anchored on the rule
of thumb of ~1,800 full crosses per skein at 14-count, scaled by fabric count
(thread per stitch goes as 1/count), rounded up per colour and never zero. They
are *not* derived from thread geometry: computing actual floss length per stitch
comes out about 1.7× optimistic, because it ignores the tail wasted starting and
finishing each thread — which dominates in a chart that changes colour
constantly. Treat the total as a floor and buy a spare of the heavily-used
colours.

## Modes

### Original design (the usual case)

No symbol key exists yet, so Chartify picks the palette itself — capped by
`--colors` — and letters it by area.

```sh
task chart -- "angel.jpg --width-stitches 200 --colors 90"
```

### Kit chart (matching a pattern you already own)

If you have the printed key, pass it and Chartify reuses that exact
symbol → DMC mapping, so the notation matches the paper chart. Threads keep
their symbol even when unused, because the notation has to stay stable against
the printed key.

```sh
task chart -- "still life.jpg --stitches 400x314 --palette palette_still_life.csv"
```

`--learn-from` goes a step further: give it a CSV of `col,row,symbol` cells
read off a real chart and it learns *that designer's* colour → thread mapping
instead of using generic DMC matching. Useful for reconstructing a damaged
chart. Requires `--palette`.

#### Palette CSV format

```csv
symbol,dmc,name,skeins
A,3328,Salmon-DK,1
Ø,310,Black,6
```

Only `symbol` and `dmc` are read. `name` overrides the built-in name if
present; `skeins` is informational. A `dmc` code absent from `dmc_colors.py` is
warned about and skipped. A duplicate `symbol` is a hard error — the mapping
has to be unambiguous.

To transcribe a new kit, read its key and write one row per thread, then spot-
check a dozen rows: reading a creased printed page is error-prone.

## Options

`task chart -- --help` lists everything. The ones that matter most:

| Flag | Does |
|---|---|
| `--stitches WxH` | Explicit stitch count, e.g. `400x314` |
| `--width-stitches N` | Stitch width; height follows the picture's aspect |
| `--width-inches N` `--count N` | Finished width in inches, at a given fabric count |
| `-c, --colors N` | Cap the palette (default 60, max 163) |
| `--squares-per-page N\|WxH` | Big squares per sheet (default 5 → 50×50 stitches) |
| `--page-size letter\|a4\|none` | Sheet to centre each page on |
| `--margin-inches N` | Binder margin down the left edge (default 0.5) |
| `--format pdf\|jpg\|png` | PDF is one booklet; jpg/png write a file per sheet |
| `--crop L,T,R,B` | Trim source pixels before charting |
| `--no-context` | Drop the faded seam strips and locator map |

The web app exposes the four that matter most — size, colours, fabric count,
paper size — and leaves the rest at their defaults.

## Running locally

```sh
task deps      # check numpy / Pillow / fonttools are present
task serve     # web app
task smoke     # generate a small chart to prove the engine works
```

Requires Python 3, NumPy and Pillow (`fonttools` is optional — it verifies
glyph coverage). `task serve` is needed rather than opening
`web/index.html` directly: the page runs its work in a web worker and fetches
the Python sources, neither of which is allowed from a `file://` origin.

## How the web app works

The browser downloads Pyodide (CPython compiled to WebAssembly) plus the NumPy
and Pillow wheels, then runs **`chartify.py` unmodified**. There is no
server-side component and nothing is uploaded — the picture is read straight
off your disk into the page.

The only adaptation is in `web/worker.js`: WebAssembly has no
`/usr/share/fonts`, so the bundled DejaVu Sans Mono is mounted into the virtual
filesystem and prepended to `chartify.FONT_CANDIDATES`. Without it Pillow falls
back to a bitmap font and every symbol becomes unreadable. Generation runs in a
web worker so a 40-page booklet doesn't freeze the page.

`web/chartify.py` and `web/dmc_colors.py` are symlinks to the copies at the
repo root — one source of truth. `task build` dereferences them into `dist/`.

### Performance

Measured in Chrome on a desktop:

| Chart | Output | Time |
|---|---|---|
| 80 × 112, 16 colours | 11-page booklet | ~5 s |
| 250 × 349, 60 colours | 40-page booklet, 87,250 stitches | ~24 s |

First visit pulls ~19 MB of Python runtime from the jsDelivr CDN, cached
thereafter. Roughly 6× slower than native, which is why the CLI still exists.

### One caveat

Pyodide currently ships Pillow 10.2.0. If your local Pillow is newer, a handful
of stitches on colour boundaries may resolve to a different thread, because the
two versions resample edge pixels slightly differently. Palette, symbols and
colour science are identical — the CIELAB conversion is bit-identical to nine
decimal places — so the charts are equally valid, just not byte-identical.

## Hosting

The site is static, so GitHub Pages serves it free from a public repo. No MIME
configuration is needed: the worker reads the `.py` files as raw bytes, so
their content type is irrelevant (verified against a server sending them as
`application/octet-stream`).

Every push to `main` touching `web/`, `chartify.py` or `dmc_colors.py`
redeploys via `.github/workflows/pages.yml`. One-time setup is
**Settings → Pages → Source → GitHub Actions**; after that, `task publish`
commits and pushes in one step.

`task build` produces a plain folder, so any static host works — Netlify,
Cloudflare Pages, S3, nginx. `task preview` serves the built output so you can
check it before uploading.

Notes: Pages needs a paid plan for *private* repos (public is free); the
workflow writes `.nojekyll` so Jekyll can't interfere; and it copies with
`cp -L` because Pages doesn't follow symlinks.

## Layout

```
chartify.py           the engine, and the CLI
dmc_colors.py         the DMC thread table
chartify              launcher, so ./chartify works
web/
  index.html          UI
  worker.js           Pyodide host, off the main thread
  vendor/             bundled font
.github/workflows/    Pages deploy
Taskfile.yml          task runner
```

Source images are gitignored — the app works on whatever picture you point it
at, so none are needed in the repo.
