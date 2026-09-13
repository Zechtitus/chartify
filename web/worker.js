/* Chart generation runs here, off the main thread, so a 40-page booklet
 * (~24s of NumPy and Pillow work) never freezes the page.
 *
 * Pyodide is loaded with importScripts rather than as an ES module because
 * the classic-worker form is the variant Pyodide documents for workers and
 * it avoids a module-worker + WASM loading path that Safari still fumbles.
 */

const PYODIDE_VERSION = "0.27.7";
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodide = null;

function post(type, payload) {
  self.postMessage({ type, ...payload });
}

/* Boot Pyodide, load the scientific stack, and mount the program plus the
 * font. Everything here is cached by the browser after the first visit. */
async function boot() {
  post("status", { stage: "runtime", message: "Downloading Python runtime…" });
  importScripts(PYODIDE_CDN + "pyodide.js");

  pyodide = await loadPyodide({ indexURL: PYODIDE_CDN });

  post("status", { stage: "packages", message: "Loading NumPy and Pillow…" });
  await pyodide.loadPackage(["numpy", "pillow", "fonttools"]);

  post("status", { stage: "program", message: "Mounting chart engine…" });

  // Fetch the program and the font from the same origin as the page.
  const [chartify, dmc, font] = await Promise.all([
    fetch("chartify.py").then(r => {
      if (!r.ok) throw new Error(`chartify.py: HTTP ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch("dmc_colors.py").then(r => {
      if (!r.ok) throw new Error(`dmc_colors.py: HTTP ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch("vendor/DejaVuSansMono-Bold.ttf").then(r => {
      if (!r.ok) throw new Error(`font: HTTP ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);

  pyodide.FS.writeFile("/chartify.py", new Uint8Array(chartify));
  pyodide.FS.writeFile("/dmc_colors.py", new Uint8Array(dmc));
  pyodide.FS.mkdirTree("/fonts");
  pyodide.FS.writeFile("/fonts/DejaVuSansMono-Bold.ttf", new Uint8Array(font));

  /* There is no /usr/share/fonts inside WASM, so load_font() would fall
   * through to PIL's tiny bitmap default and every symbol would be
   * unreadable. load_font is the single chokepoint for font lookup, so
   * prepending the mounted path is the whole fix -- chartify.py itself is
   * used unmodified. */
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/")
import chartify
from PIL import ImageFont

chartify.FONT_CANDIDATES = ("/fonts/DejaVuSansMono-Bold.ttf",) + chartify.FONT_CANDIDATES

# Fail loudly now rather than silently emitting an unreadable chart later.
_probe = chartify.load_font(20)
if not isinstance(_probe, ImageFont.FreeTypeFont):
    raise RuntimeError("mounted font did not load as TrueType")
`);

  post("ready", {});
}

const bootPromise = boot().catch(err => {
  post("fatal", { message: String(err && err.message || err) });
  throw err;
});

/* Run one chart. Options arrive already validated by the UI. */
async function generate(opts) {
  await bootPromise;

  const name = opts.filename || "input.png";
  // Keep the real extension: Pillow picks its decoder from the bytes, but a
  // sensible name makes the cover sheet's "source" line meaningful.
  const ext = (name.match(/\.[A-Za-z0-9]+$/) || [".png"])[0];
  pyodide.FS.writeFile("/input" + ext, new Uint8Array(opts.imageBytes));

  // Clear any previous run so stale files can't be mistaken for new output.
  await pyodide.runPythonAsync(`
import shutil, os
shutil.rmtree("/out", ignore_errors=True)
os.makedirs("/out", exist_ok=True)
`);

  post("status", { stage: "charting", message: "Matching colours to DMC threads…" });

  // Build argv in Python to sidestep any JS/Python string-quoting concerns.
  pyodide.globals.set("js_argv", pyodide.toPy([
    "/input" + ext,
    "--width-stitches", String(opts.widthStitches),
    "--colors", String(opts.colors),
    "--count", String(opts.count),
    "--page-size", opts.pageSize,
    "-o", "/out",
    "--format", "pdf",
  ]));

  const resultJson = await pyodide.runPythonAsync(`
import io, json, os, sys, time
import chartify

_buf = io.StringIO()
_stdout = sys.stdout
sys.stdout = _buf
_started = time.time()
try:
    rc = chartify.main(list(js_argv))
finally:
    sys.stdout = _stdout

log = _buf.getvalue()
if rc != 0:
    raise RuntimeError("chart generation failed (exit %s)\\n%s" % (rc, log[-800:]))

files = []
for entry in sorted(os.listdir("/out")):
    files.append({"name": entry, "size": os.path.getsize("/out/" + entry)})

json.dumps({
    "rc": rc,
    "seconds": round(time.time() - _started, 1),
    "log": log,
    "files": files,
})
`);

  const result = JSON.parse(resultJson);

  // Hand the bytes back as transferable buffers so nothing is copied twice.
  const transfers = [];
  for (const f of result.files) {
    const bytes = pyodide.FS.readFile("/out/" + f.name);
    // Copy into a standalone buffer; the FS view is backed by WASM memory.
    const buf = new Uint8Array(bytes).buffer;
    f.buffer = buf;
    transfers.push(buf);
  }

  self.postMessage({ type: "done", result }, transfers);
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type !== "generate") return;
  try {
    await generate(msg);
  } catch (err) {
    post("error", { message: String(err && err.message || err) });
  }
};
