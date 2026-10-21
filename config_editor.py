#!/usr/bin/env python3
"""
Local web form for editing pipeline_config.yaml's AreTomo3/IsoNet2
parameters and raw data paths, without hand-editing YAML.

Usage:
    python3 config_editor.py
    python3 config_editor.py --config pipeline_config.yaml --port 8765

Then open http://localhost:8765 in a browser. Over SSH/VSCode Remote, the
port is auto-forwarded -- no extra setup needed. The server binds to
127.0.0.1 only (not reachable from the network) since it can rewrite the
config file.

This edits pipeline_config.yaml IN PLACE by patching only the matched
value text of each field's own line -- it never re-serializes the whole
file with a YAML dumper, so every comment (including the AreTomo3-segfault
and epochs/save_interval gotcha notes) survives untouched.

Scope: only fields already active (uncommented) in the YAML are editable
here. Commented-out advanced options (predict.model, refine.pretrained_model,
organize.pixel_size override, prepare_star.number_subtomos, the aretomo
"extra:" passthrough) still need a manual edit -- editing those safely would
require deciding how to (re)comment a line, which this tool intentionally
does not guess at.

This does NOT run the pipeline -- it only edits the config. Run
run_pipeline.py yourself afterward, as before.
"""
import argparse
import html
import re
import yaml
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

# (dotted_path, label, kind) or (dotted_path, label, kind, options) -- kind
# is one of: str, path, int, float, bool, intlist, str_or_null, select
# (options required for select: a list of (value, display_label) pairs).
#
# SECTIONS entries are (title, fields, engine_filter) -- engine_filter is
# None for a section that's always shown, or one of "aretomo3" / "aretomo2"
# / "imod_raptor" for a section the page only shows while the top-level
# "engine" field is set to that value (see render_form's <script> for the
# show/hide logic).
SECTIONS = [
    ("Engine selection", [
        ("engine", "Alignment engine", "select",
         [("aretomo3", "AreTomo3 (raw frames + .mdoc)"),
          ("aretomo2", "AreTomo2 (pre-assembled tilt series .mrc)"),
          ("imod_raptor", "IMOD + RAPTOR (fiducial-based)")]),
        # isonet_bin is deliberately NOT editable here: it can be either a
        # plain string or a [python_path, script_path] list (see the
        # comment above it in the YAML), and this editor's "str" kind
        # would corrupt a list value by quoting its Python repr on save.
        # Edit it by hand.
    ], None),
    ("AreTomo3 -- raw paths", [
        ("aretomo_bin", "AreTomo3 binary", "str"),
        ("aretomo.in_prefix", "Input frames/mdoc dir", "path"),
        ("aretomo.out_dir", "Output dir", "path"),
        ("aretomo.tmp_dir", "Temp dir", "path"),
        ("aretomo.log_dir", "Log dir", "path"),
        ("aretomo.gain", "Gain reference file", "path"),
    ], "aretomo3"),
    ("AreTomo3 -- parameters", [
        ("aretomo.in_suffix", "Input suffix", "str"),
        ("aretomo.in_skips", "Skip marker", "str"),
        ("aretomo.pix_size", "Raw pixel size (A)", "float"),
        ("aretomo.kv", "Voltage (kV)", "float"),
        ("aretomo.cs", "Spherical aberration (mm)", "float"),
        ("aretomo.amp_contrast", "Amplitude contrast", "float"),
        ("aretomo.fm_dose", "Per-frame dose", "float"),
        ("aretomo.eer_sampling", "EER sampling (pair with mc_bin=2 for super-resolution)", "int"),
        ("aretomo.mc_bin", "Motion-cor Fourier binning", "float"),
        ("aretomo.group", "Frame grouping (global patch)", "intlist"),
        ("aretomo.mc_patch", "Motion-cor patches (x y)", "intlist"),
        ("aretomo.tilt_axis", "Tilt axis", "intlist"),
        ("aretomo.tilt_cor", "Tilt offset correction", "int"),
        ("aretomo.align_z", "Alignment Z height (specimen thickness, px)", "int"),
        ("aretomo.vol_z", "Volume Z height", "int"),
        ("aretomo.at_bin", "AtBin (primary secondary)", "intlist"),
        ("aretomo.wbp", "WBP reconstruction", "int"),
        ("aretomo.corr_ctf", "CTF correction", "int"),
        ("aretomo.out_imod", "OutImod mode", "int"),
        ("aretomo.split_sum", "Split odd/even sums", "int"),
        ("aretomo.resume", "Resume", "int"),
        ("aretomo.gpu", "GPU IDs", "intlist"),
    ], "aretomo3"),
    ("AreTomo2 (pre-assembled tilt series .mrc)", [
        ("aretomo2_bin", "AreTomo2 binary", "str"),
        ("aretomo2.in_dir", "Input dir (pre-assembled tilt series .mrc)", "path"),
        ("aretomo2.out_dir", "Output dir", "path"),
        ("aretomo2.out_suffix", "Output filename suffix", "str"),
        ("aretomo2.skip_existing", "Skip files whose output already exists", "bool"),
        ("aretomo2.pix_size", "Pixel size (A)", "float"),
        ("aretomo2.kv", "Voltage (kV)", "float"),
        ("aretomo2.cs", "Spherical aberration (mm)", "float"),
        ("aretomo2.amp_contrast", "Amplitude contrast", "float"),
        ("aretomo2.tilt_range", "Tilt range (min max)", "intlist"),
        ("aretomo2.tilt_axis", "Tilt axis (angle, -1=use as-is)", "intlist"),
        ("aretomo2.align_z", "Alignment Z height (specimen thickness, px)", "int"),
        ("aretomo2.vol_z", "Volume Z height", "int"),
        ("aretomo2.out_bin", "Output binning", "float"),
        ("aretomo2.wbp", "WBP reconstruction", "int"),
        ("aretomo2.patch", "Local alignment patches (x y)", "intlist"),
        ("aretomo2.gpu", "GPU IDs", "intlist"),
    ], "aretomo2"),
    ("IMOD + RAPTOR -- raw paths / input", [
        ("imod.bin_dir", "IMOD bin/ dir (e.g. /usr/local/imod_5.1.0/bin)", "path"),
        ("imod.input_mode", "Input mode", "select",
         [("eer", "eer (raw EER frames + .mdoc, via alignframes)"),
          ("mrc", "mrc (pre-assembled tilt series .st/.mrc)")]),
        ("imod.in_dir", "Input dir (EER frames+.mdoc, or .mrc stacks if input_mode=mrc)", "path"),
        ("imod.in_suffix", "Input suffix (input_mode=eer)", "str"),
        ("imod.gain", "Gain reference file (input_mode=eer)", "path"),
        ("imod.out_dir", "Output dir", "path"),
    ], "imod_raptor"),
    ("IMOD + RAPTOR -- parameters", [
        ("imod.voltage", "Voltage (kV, 200 or 300 only)", "int"),
        ("imod.pixel_size_angstrom", "Raw pixel size (A)", "float"),
        ("imod.bin_factor", "Bin factor before RAPTOR", "int"),
        ("imod.thickness_px", "Reconstructed tomogram thickness (px)", "int"),
        ("imod.gold_diameter_nm", "Gold fiducial diameter (nm)", "float"),
        ("imod.out_suffix", "Final tomogram filename suffix", "str"),
        ("imod.feed_variant", "Variant fed downstream (wbp | fake_sirt | filtered)", "str"),
        ("imod.xray_removal.enabled", "Remove X-rays/hot pixels (ccderaser) before binning", "bool"),
        ("imod.erase_gold.enabled", "Erase gold fiducials after alignment (before reconstruction)", "bool"),
        ("imod.erase_gold.better_radius_px", "Gold erase radius override (px, blank=default)", "str_or_null"),
        ("imod.raptor.markers_per_image", "Markers to track per image (blank=auto-estimate)", "str_or_null"),
        ("imod.raptor.xray", "RAPTOR: remove strong line features before tracking", "bool"),
        ("imod.reconstruction.fake_sirt_iterations", "FakeSIRT iterations (blank=skip this variant)", "str_or_null"),
        ("imod.filtered_reconstruction.enabled", "Also reconstruct an mtffilter-filtered variant", "bool"),
    ], "imod_raptor"),
    ("Organize (-> IsoNet2 layout)", [
        ("organize.out_dir", "Organized tomogram output dir", "path"),
        ("organize.raw_pixel_size", "Raw pixel size (A) -- aretomo3 only", "float"),
        ("organize.bin_factor", "Bin factor -- aretomo3 only, must match aretomo.at_bin[0]", "float"),
        ("organize.defocus_source", "Defocus source (ctf | none) -- aretomo3 only", "str"),
        ("organize.copy", "Copy instead of symlink", "bool"),
    ], None),
    ("IsoNet2 -- prepare_star", [
        ("isonet.prepare_star.star_name", "Star filename", "str"),
        ("isonet.prepare_star.voltage", "Voltage (kV)", "float"),
        ("isonet.prepare_star.ac", "Amplitude contrast", "float"),
        ("isonet.prepare_star.tilt_min", "Tilt min", "float"),
        ("isonet.prepare_star.tilt_max", "Tilt max", "float"),
    ], None),
    ("IsoNet2 -- deconv (usually disabled for even/odd data -- see comment in file)", [
        ("isonet.deconv.enabled", "Enabled", "bool"),
        ("isonet.deconv.deconvstrength", "Deconv strength", "float"),
        ("isonet.deconv.snrfalloff", "SNR falloff", "float"),
        ("isonet.deconv.highpassnyquist", "High-pass (fraction of Nyquist)", "float"),
        ("isonet.deconv.phaseflipped", "Phase-flipped input", "bool"),
        ("isonet.deconv.ncpus", "CPUs", "int"),
        ("isonet.deconv.output_dir", "Output dir", "str"),
    ], None),
    ("IsoNet2 -- make_mask", [
        ("isonet.make_mask.density_percentage", "Density percentage", "int"),
        ("isonet.make_mask.std_percentage", "Std percentage", "int"),
        ("isonet.make_mask.patch_size", "Patch size", "int"),
        ("isonet.make_mask.z_crop", "Z crop fraction", "float"),
        ("isonet.make_mask.output_dir", "Output dir", "str"),
    ], None),
    ("IsoNet2 -- refine", [
        ("isonet.refine.method", "Method (isonet2 | isonet2-n2n)", "str"),
        ("isonet.refine.arch", "Architecture", "str"),
        ("isonet.refine.cube_size", "Cube size (multiple of 16)", "int"),
        ("isonet.refine.epochs", "Epochs (must be >= save_interval, see comment)", "int"),
        ("isonet.refine.learning_rate", "Learning rate", "str"),
        ("isonet.refine.loss_func", "Loss function", "str"),
        ("isonet.refine.CTF_mode", "CTF mode (blank=default | phase_only | wiener | network)", "str_or_null"),
        ("isonet.refine.bfactor", "B-factor", "float"),
        ("isonet.refine.mw_weight", "Missing-wedge weight (-1 disables)", "float"),
        ("isonet.refine.noise_level", "Noise level", "float"),
        ("isonet.refine.mixed_precision", "Mixed precision", "bool"),
        ("isonet.refine.gpuID", "GPU IDs (e.g. 0,1,2,3)", "str"),
        ("isonet.refine.ncpus", "CPUs", "int"),
        ("isonet.refine.save_interval", "Save interval (epochs must be a multiple of this)", "int"),
        ("isonet.refine.output_dir", "Output dir", "str"),
    ], None),
    ("IsoNet2 -- predict", [
        ("isonet.predict.apply_mw_x1", "Apply missing-wedge mask", "bool"),
        ("isonet.predict.isCTFflipped", "Input already phase-flipped", "bool"),
        ("isonet.predict.padding_factor", "Padding factor", "float"),
        ("isonet.predict.gpuID", "GPU IDs", "str"),
        ("isonet.predict.output_dir", "Output dir", "str"),
    ], None),
    ("RELION5 star export (optional, works with any engine)", [
        ("relion5_star.out_dir", "Output dir", "path"),
        ("relion5_star.defocus_handedness", "Defocus handedness (aretomo2/imod_raptor only)", "int"),
    ], None),
    ("PyTom -- template matching (optional)", [
        ("pytom.bin_dir", "pytom_tm conda env's bin/ dir", "path"),
        ("pytom.reference_map", "Reference density map (MRC)", "path"),
        ("pytom.particle_diameter_angstrom", "Particle diameter (A)", "float"),
        ("pytom.voxel_size_angstrom", "Voxel size (A) -- must match predict's REAL output pixel size, not its MRC header", "float"),
        ("pytom.out_dir", "Output dir", "path"),
        ("pytom.gpu", "GPU IDs", "intlist"),
        ("pytom.create_mask.sigma", "Mask edge drop-off (px)", "float"),
        ("pytom.match_template.tilt_angles", "Tilt angles (min max)", "intlist"),
        ("pytom.extract_candidates.number_of_particles", "Max particles to extract per tomogram", "int"),
        ("pytom.extract_candidates.relion5-compat", "RELION5-compatible coordinates (particles.star)", "bool"),
    ], None),
]

KEY_RE = re.compile(r'^(?P<indent>[ \t]*)(?P<key>"[^"]+"|[A-Za-z0-9_.\-]+):(?P<rest>.*)$')
VALUE_RE = re.compile(r'^(?P<prefix>[ \t]*(?:"[^"]+"|[A-Za-z0-9_.\-]+):[ \t]*)(?P<value>[^#]*?)(?P<suffix>[ \t]*(?:#.*)?)$')


def get_by_path(cfg: dict, path: str):
    node = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def index_paths(lines: list) -> dict:
    """Map each dotted key path to its line index, by tracking an
    indentation stack. Comment-only and blank lines are ignored so they
    never affect nesting."""
    stack = []  # list of (indent, key)
    index = {}
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = KEY_RE.match(line)
        if not m:
            continue
        indent = len(m.group("indent").expandtabs())
        key = m.group("key").strip('"')
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([k for _, k in stack] + [key])
        stack.append((indent, key))
        index[path] = i
    return index


def format_value(kind: str, raw: str) -> str:
    """Turn a form field's raw text into the literal YAML text to place
    after the colon on its line."""
    raw = raw.strip()
    if kind == "bool":
        return "true" if raw else "false"
    if kind == "str_or_null":
        if raw == "":
            return "null"
        escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if kind == "intlist":
        parts = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
        nums = []
        for p in parts:
            f = float(p)
            nums.append(str(int(f)) if f.is_integer() else str(f))
        return "[" + ", ".join(nums) + "]"
    if kind == "int":
        return str(int(float(raw)))
    if kind == "float":
        float(raw)  # validate
        return raw
    # str, path -- always double-quote so arbitrary user input (colons,
    # commas, leading #/*/&/!, etc.) can never be misparsed as something
    # other than a plain string.
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def display_value(kind: str, value) -> str:
    if value is None:
        return ""
    if kind == "bool":
        return "checked" if value else ""
    if kind == "intlist" and isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def render_form(cfg: dict, message: str = "", error: str = "") -> str:
    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>Pipeline config editor</title>
<style>
body{font-family:-apple-system,Segoe UI,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#faf9f5;color:#1a1a1a}
h1{font-size:1.4rem} h2{font-size:1.05rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
.field{display:flex;align-items:center;gap:.75rem;margin:.4rem 0}
.field label{flex:0 0 380px;font-size:.88rem;color:#333}
.field input[type=text],.field select{flex:1;padding:.3rem .5rem;font-family:ui-monospace,monospace;font-size:.85rem}
.field input[type=checkbox]{width:1.1rem;height:1.1rem}
.msg{padding:.6rem 1rem;border-radius:6px;margin:1rem 0}
.msg.ok{background:#e2f4e2;color:#175417}
.msg.err{background:#fbe2e2;color:#7a1414}
button{padding:.6rem 1.4rem;font-size:.95rem;margin-top:1.5rem;cursor:pointer}
.note{color:#666;font-size:.8rem;margin-top:.3rem}
section[hidden]{display:none}
</style></head><body>
<h1>AreTomo3 / AreTomo2 / IMOD+RAPTOR / IsoNet2 pipeline config</h1>
<p class="note">Editing this file's fields directly -- comments and structure are preserved. Advanced/commented-out options (predict.model, refine.pretrained_model, the aretomo "extra:" block, etc.) still need a manual edit. Sections for the two engines you didn't pick above are hidden, not deleted -- their values stay in the file untouched.</p>
"""]
    if message:
        parts.append(f'<div class="msg ok">{html.escape(message)}</div>')
    if error:
        parts.append(f'<div class="msg err">{html.escape(error)}</div>')
    parts.append('<form method="post" action="/save">')
    for section_title, fields, engine_filter in SECTIONS:
        attr = f' data-engine="{html.escape(engine_filter)}"' if engine_filter else ""
        parts.append(f"<section{attr}><h2>{html.escape(section_title)}</h2>")
        for field in fields:
            path, label, kind = field[0], field[1], field[2]
            value = get_by_path(cfg, path)
            name = html.escape(path)
            lbl = html.escape(label)
            if kind == "bool":
                checked = "checked" if value else ""
                parts.append(
                    f'<div class="field"><label for="{name}">{lbl}</label>'
                    f'<input type="checkbox" id="{name}" name="{name}" {checked}></div>'
                )
            elif kind == "select":
                options = field[3]
                opts_html = "".join(
                    f'<option value="{html.escape(v)}"'
                    f'{" selected" if str(value) == v else ""}>{html.escape(lbl_opt)}</option>'
                    for v, lbl_opt in options
                )
                parts.append(
                    f'<div class="field"><label for="{name}">{lbl}</label>'
                    f'<select id="{name}" name="{name}">{opts_html}</select></div>'
                )
            else:
                val = html.escape(display_value(kind, value))
                parts.append(
                    f'<div class="field"><label for="{name}">{lbl}</label>'
                    f'<input type="text" id="{name}" name="{name}" value="{val}"></div>'
                )
        parts.append("</section>")
    parts.append('<button type="submit">Save to pipeline_config.yaml</button></form>')
    parts.append("""<script>
(function(){
  var sel = document.getElementById("engine");
  if (!sel) return;
  function apply(){
    var chosen = sel.value;
    document.querySelectorAll("section[data-engine]").forEach(function(s){
      s.hidden = s.getAttribute("data-engine") !== chosen;
    });
  }
  sel.addEventListener("change", apply);
  apply();
})();
</script></body></html>""")
    return "".join(parts)


def all_fields():
    for _, fields, _ in SECTIONS:
        for field in fields:
            yield field[0], field[1], field[2]


def apply_updates(config_path: Path, form: dict) -> None:
    text = config_path.read_text()
    lines = text.split("\n")
    index = index_paths(lines)

    for path, label, kind in all_fields():
        if kind == "bool":
            raw = "1" if path in form else ""
        else:
            raw = form.get(path, [""])[0]
        new_value_text = format_value(kind, raw)

        if path not in index:
            raise ValueError(
                f"'{label}' ({path}) isn't an active (uncommented) line in "
                f"{config_path.name} -- this editor can't add new lines, "
                f"only edit existing ones. Uncomment it manually first."
            )
        line_idx = index[path]
        m = VALUE_RE.match(lines[line_idx])
        if not m:
            raise ValueError(f"couldn't parse existing line for '{label}': {lines[line_idx]!r}")
        lines[line_idx] = m.group("prefix") + new_value_text + m.group("suffix")

    config_path.write_text("\n".join(lines))


class Handler(BaseHTTPRequestHandler):
    config_path: Path = None  # set by main()

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet

    def _send_html(self, body: str, status: int = 200):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path not in ("/", ""):
            self._send_html("Not found", 404)
            return
        cfg = yaml.safe_load(self.config_path.read_text())
        self._send_html(render_form(cfg))

    def do_POST(self):
        if self.path != "/save":
            self._send_html("Not found", 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body, keep_blank_values=True)
        try:
            apply_updates(self.config_path, form)
            cfg = yaml.safe_load(self.config_path.read_text())
            self._send_html(render_form(cfg, message=f"Saved to {self.config_path}"))
        except Exception as exc:
            cfg = yaml.safe_load(self.config_path.read_text())
            self._send_html(render_form(cfg, error=f"Not saved -- {exc}"), status=400)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("pipeline_config.yaml"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    args = ap.parse_args()

    if not args.config.exists():
        raise SystemExit(f"{args.config} not found -- copy pipeline_config.example.yaml first")

    Handler.config_path = args.config
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Editing {args.config}")
    print(f"Open http://localhost:{args.port} in your browser (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
