# Chartify — working notes

Conventions and hard-won constraints for this repo. Read this before changing
the engine; several of the rules below exist because the obvious approach was
tried and failed.

## What this is

A cross-stitch chart generator with two front ends over one engine:

- `chartify.py` — the engine *and* the CLI. ~2,000 lines, no framework.
- `web/` — a static page that runs the same `chartify.py` in the browser under
  Pyodide (CPython on WebAssembly).

`web/chartify.py` and `web/dmc_colors.py` are **symlinks** to the root copies.
There is one source of truth; never create a second copy. `task build`
dereferences them with `cp -L` into `dist/`.

## Golden rule: the engine stays a single file with no new dependencies

The engine must keep running unmodified under Pyodide. That rules out:

- **New third-party dependencies.** Only NumPy, Pillow and (optionally)
  fontTools are available. No pypdf, pikepdf, reportlab, fpdf2 or img2pdf in
  Pyodide — I checked. Anything else means a micropip download at page load.
- **Filesystem assumptions.** There is no `/usr/share/fonts` in WASM. Font
  lookup goes through `load_font()`, which is the single chokepoint — the web
  worker prepends its bundled font path to `FONT_CANDIDATES` and changes
  nothing else. Keep it that way.
- **Anything platform-specific** — subprocesses, threads, sockets.

If a change would need a dependency, solve it in stdlib instead (see the PDF
writer below).

## Memory is the binding constraint

A Letter sheet at 300dpi is 2550×3300 RGB — about **25MB per page**. The
browser heap is far smaller than a desktop's, and overruns surface as
`MemoryError` deep inside Pillow, usually in a `resize`.

Rules that follow from this:

- **Never accumulate rendered pages in a list.** Pages are spooled to disk in
  `outdir/.pages` as they are finished, and the spool is cleaned up in a
  `finally`.
- **Pillow's multi-page PDF writer cannot stream.** `save(save_all=True,
  append_images=[...])` decodes and holds *every* page for the whole write:
  1,311MB for 40 sheets. Spooling to disk does not help; pre-encoding the
  pages as JPEG does not help; closing pages during iteration crashes because
  Pillow re-reads them while encoding. All three were tried.
- So `write_pdf_streaming()` writes the PDF container by hand — one
  JPEG-compressed `/DCTDecode` image XObject per page, encoded and released
  before the next page is read. Peak drops to ~87MB. Past
  `PILLOW_PAGE_LIMIT` (12) pages this is the default path; smaller booklets
  still use Pillow's writer, with a `MemoryError` fallback.
- Peak RSS for a 40-page chart: **~435MB**, down from ~1,488MB. If you change
  the render or save path, re-measure with
  `/usr/bin/time -v` and keep it in that range.

## Colour science is correctness-critical, and silent when wrong

- Matching runs in **CIELAB with CIEDE2000**, never RGB-Euclidean. RGB
  matching picks visibly wrong threads, worst in skin tones and muted greens.
- `ciede2000()` is verified against the **Sharma et al. reference set** (22
  vectors in the tests) to `abs=2e-4`. Do not loosen that tolerance: dropping
  the hue-wraparound term shifts one reference case by only 7e-4, so a `1e-3`
  tolerance lets a real bug through. This was found by mutation testing.
- A wrong thread match never raises. It is only visible once someone has
  stitched half the piece, which is why the tests pin the numbers.

## Symbol alphabet

163 glyphs, ordered roughly by legibility at print size; the commonest colour
gets the clearest symbol. Deliberately **excludes** whole families of
lookalikes — box-drawing pieces, filled/hollow circle and triangle sets, dice
faces — because they read as the same mark in a cell a few millimetres across.
Do not "helpfully" add them back to raise the colour cap.

Every glyph must exist in the bundled font. `_covers_alphabet()` verifies this
when fontTools is present.

## Output conventions

- Front matter is numbered `00a`–`00e` so a plain filename sort collates the
  booklet: cover, preview, shopping list, key, page map, then `page01`onward.
- Tile origin is the **top-left**; `x` increases right, `y` increases down,
  both 1-based. `chart-x1y1` is top-left.
- Pages are sized in **big squares**, not pixels, so a sheet is always a whole
  number of squares and the grid never splits across a seam.
- Chart pages carry two stitches of each adjoining page down their edges,
  greyed and offset by a gap, so notation can be matched across a seam.
- Skein estimates are anchored on **~1,800 crosses per skein at 14-count**,
  scaled as 1/count, rounded up, never zero. Deliberately *not* derived from
  thread geometry — that comes out ~1.7× optimistic because it ignores the
  tail wasted starting and finishing each thread. Treat it as a floor.

## Testing

`task unittest` (or `task test`). 63 tests: ~12s across 12 cores, ~32s
serial. Every test name is printed, with the five slowest summarised.

Parallelism comes from pytest-xdist (`sudo apt install
python3-pytest-xdist`; the system Python is PEP 668 externally-managed, so
don't `pip install` into it). The task detects xdist and falls back to
serial without it.

Use plain `-n auto`, **not** `--dist loadfile`: the whole suite is one file,
so loadfile schedules everything onto a single worker and saves nothing
(measured 33s either way, versus 12s distributing per test). Workers are
separate processes, so the `ru_maxrss` assertions in the memory tests stay
per-worker and are safe under parallelism.

What the suite is for: the things that would **corrupt every chart without
crashing**. Colour science against reference data, page geometry (a sheet that
prints the wrong physical size is useless), the parsers, and both PDF paths.

- Rendering is tested for **invariants** — size, page count, ordering,
  determinism — not pixel-compared against reference images. Reference images
  would break on every font or Pillow upgrade without indicating a fault.
- `test_large_booklet_does_not_collate_in_memory` guards the specific
  regression that broke the browser. It asserts on *which writer runs*, not on
  RSS, because RSS through `main()` is too noisy to assert on.
- When adding a test, check it actually fails when you break the code.
  Several of these tests passed against deliberately broken code on the first
  attempt and had to be tightened.

Verify engine changes two ways:
1. `task unittest`
2. A pixel diff against the previous output — render both to PNG with
   `pdftoppm -r 40 -png` and compare hashes. Output should be pixel-identical
   unless the change is meant to alter it. Note the PDF itself embeds a
   timestamp, so compare *rendered pages*, not PDF bytes.

## Web app

- Generation runs in `web/worker.js`, off the main thread, so a 100-page
  booklet does not freeze the page.
- The worker fetches `chartify.py` as **raw bytes** (`arrayBuffer`), so its
  served Content-Type is irrelevant. GitHub Pages serves it as
  `application/octet-stream`; verified this works, so no host MIME config is
  needed.
- Pyodide is pinned to **v0.27.7** (Python 3.12.7, NumPy 2.0.2, Pillow
  10.2.0) and loaded from jsDelivr, so it costs no Pages bandwidth. Do not
  use a `.dev0` build.
- Pyodide's Pillow is older than a current desktop's. Boundary stitches can
  resolve to a different thread because the two versions resample edge pixels
  differently. The colour science is bit-identical; only resampling differs.
  Expect near-identical, not byte-identical, output between CLI and web.
- The UI exposes four options only — width, colours, fabric count, paper size.
  Everything else stays at CLI defaults. Keep it that way unless asked.

## Deployment

- GitHub Pages via `.github/workflows/pages.yml`, on push to `main` touching
  `web/`, `chartify.py` or `dmc_colors.py`. Live at
  <https://zechtitus.github.io/chartify/>.
- Pages does **not** follow symlinks — hence `cp -L` in the workflow. The
  workflow also asserts nothing in `dist/` is still a symlink.
- All actions are pinned to **Node 24 majors** (`checkout@v7`,
  `configure-pages@v6`, `upload-pages-artifact@v5`, `deploy-pages@v5`). Note
  `upload-pages-artifact` is a composite: v3 called `upload-artifact@v4`
  (Node 20) internally, so bumping only the outer version does not silence the
  deprecation warning.
- `.nojekyll` is written so Jekyll cannot interfere with the assets.

## Repo hygiene

- **Source images are gitignored** (`*.jpg`, `*.jpeg`, `*.png`, with
  `!web/favicon.png`). The repo is public; it held a personal photo and some
  third-party artwork, and the app needs no images of its own. Do not commit
  test images — generate them synthetically, as the tests do.
- Generated chart folders (`*-chart-*/`) are gitignored; a single run can be
  tens of MB.
- Commits here use a GitHub noreply address, set repo-locally. Don't change
  the global git identity.
