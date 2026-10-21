#!/usr/bin/env python3
"""
Fiducial-based tilt series alignment and reconstruction via IMOD + RAPTOR
(imod_raptor engine, conda-free install at imod.bin_dir -- confirmed real,
not a conda env, on this machine: /usr/local/imod_5.1.0/bin).

This is a THIRD, independent engine alongside AreTomo3/AreTomo2 -- it does
NOT route through AreTomo for motion correction. Two separate input modes,
selected by imod.input_mode:

  "eer": raw EER frames + a companion .mdoc per tilt series (same in_dir/
    in_suffix convention as aretomo.in_prefix). Motion correction and
    per-tilt frame summing is done by IMOD's OWN alignframes tool (NOT
    AreTomo) -- confirmed via --help: alignframes natively supports EER
    (-eer), gain reference (-gain), reading grouping straight from the
    .mdoc (-mdoc/-path), and writing a combined multi-section output
    stack (-stack) with tilt angle embedded in each section's header
    (-tilt) -- exactly what extracttilts (next step) reads back out.

  "mrc": one pre-assembled tilt series stack per position (*.st/*.mrc,
    already motion-corrected elsewhere), same as the original bin_tomo.sh
    script this module is based on assumed. Tilt angle must already be in
    each section's header (standard for a real SerialEM-collected .st).

From there, both modes converge onto the same chain, matching the
originally-working bin_tomo.sh script (GAF, 2018) step for step, with all
flags re-verified against this install's real --help / man page text on
2026-10-21:

  extracttilts (header mode) -> ccderaser -find (x-ray/hot-pixel removal)
  -> newstack -multadd/-bin/-antialias (shift above 0 + bin)
  -> RAPTOR (fiducial tracking + alignment; -bin 1 since the stack is
     already pre-binned, diameter given in THIS stack's own pixel size --
     matches the original script's math and RAPTOR's own man-page caveat
     that diameter must be given "in the original image indicated by
     -input", which here IS the pre-binned stack)
  -> [optional] erase_gold(): point2model -scat on RAPTOR's own
     IMOD/*.fid.txt fiducial model, then ccderaser -model ... -circle to
     erase the tracked gold beads from the aligned stack before
     reconstruction (the user's own requested feature -- RAPTOR's man
     page confirms *.fid.txt is the right file: "IMOD folder: contains a
     final fiducial model (*.fid.txt)")
  -> tilt -RotateBy90 -THICKNESS ... (WBP reconstruction), optionally also
     with -FakeSIRTiterations, and optionally a second reconstruction from
     an mtffilter-filtered copy of the aligned stack.

This module is a direct, literal translation of bin_tomo.sh step for
step -- same flags, same -mode 1 binning, same mv-based promotion of
RAPTOR's output files, no added retries or workarounds. RAPTOR needs
real gold fiducial markers in the sample.
"""
import math
import os
import shutil
import subprocess
from pathlib import Path


def _env(cfg: dict):
    """RAPTOR (and possibly other IMOD tools) internally call sibling
    IMOD programs (e.g. RAPTOR calls tiltalign) by bare name, relying on
    PATH -- confirmed from a real failure: "ERROR: error executing the
    command tiltalign -param ... in function RAPTOPR::Main" when PATH
    (inherited from whatever launched run_pipeline.py) didn't include
    IMOD's bin dir. So PATH/LD_LIBRARY_PATH are always set explicitly
    here from cfg.bin_dir, the same way IMOD's own RAPTOR wrapper script
    sets LD_LIBRARY_PATH for itself -- not relying on the caller's shell
    environment at all."""
    bin_dir = cfg.get("bin_dir") if cfg else None
    if not bin_dir:
        return None
    imod_dir = Path(bin_dir).parent
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{imod_dir}/qtlib:{imod_dir}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def _run(cmd: list, dry_run: bool, cfg: dict = None):
    print("+ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, env=_env(cfg))


def _bin(cfg: dict, name: str) -> str:
    bin_dir = cfg.get("bin_dir")
    return str(Path(bin_dir) / name) if bin_dir else name


def gold_diameter_px(cfg: dict) -> int:
    """Diameter of the gold markers in pixels of the BINNED stack RAPTOR
    is actually given (RAPTOR gets -bin 1, i.e. no further binning of its
    own -- the pre-binning happens upstream via newstack). Matches
    bin_tomo.sh's own rounding formula exactly."""
    gold_nm = cfg["gold_diameter_nm"]
    px_a = cfg["pixel_size_angstrom"]
    bin_factor = cfg["bin_factor"]
    return round((gold_nm * 10 + px_a * bin_factor / 2) / (px_a * bin_factor))


def align_frames_eer(cfg: dict, mdoc_path: Path, out_dir: Path, dry_run: bool) -> tuple:
    """imod.input_mode: eer. Motion-correct + sum raw EER frames for one
    tilt series into a single multi-section stack, tilt angle embedded
    per-section (so the next step, extracttilts, can read it back out of
    the header exactly like it would for a real pre-assembled .st).
    Returns (stack_path, base_name)."""
    base = mdoc_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stack_path = out_dir / f"{base}_assembled.st"
    angles_path = out_dir / f"{base}_mdoc.rawtlt"

    # Bootstrap the angle file from the .mdoc directly (extracttilts
    # supports -mdoc for exactly this -- no image header exists yet since
    # nothing has been assembled from the raw frames at this point).
    _run([_bin(cfg, "extracttilts"), "-mdoc", str(mdoc_path), str(angles_path)], dry_run, cfg)

    eer_params = cfg.get("eer_params", [1, 10, -1])
    cmd = [_bin(cfg, "alignframes"),
           "-mdoc", str(mdoc_path),
           "-path", str(cfg["in_dir"]),
           "-gain", str(cfg["gain"]),
           "-eer", ",".join(str(v) for v in eer_params)]
    cmd += ["-volt", str(cfg.get("voltage", 300)),
            "-stack", str(stack_path),
            "-tilt", str(angles_path),
            "-reorder", "1"]
    cmd += _extra_args(cfg.get("align_frames", {}))
    _run(cmd, dry_run, cfg)
    return stack_path, base


def extract_rawtlt(cfg: dict, stack: Path, out_name: Path, dry_run: bool) -> Path:
    """extracttilts (header mode): pulls tilt angle back out of the
    stack's own section headers. Matches bin_tomo.sh's first step exactly
    (it names the output using the FUTURE binned stack's basename even
    though extraction runs on the pre-binned file -- tilt angle doesn't
    change with binning, only the filename needs to match later)."""
    _run([_bin(cfg, "extracttilts"), str(stack), str(out_name)], dry_run, cfg)
    return out_name


def erase_xrays(cfg: dict, stack: Path, out_dir: Path, dry_run: bool) -> Path:
    base = stack.stem
    out_stack = out_dir / f"{base}_NoXray.st"
    out_model = out_dir / f"{base}_NoXray.mod"
    xr_cfg = dict(cfg.get("xray_removal", {}))
    xr_cfg.pop("enabled", None)
    cmd = [_bin(cfg, "ccderaser"), "-input", str(stack), "-output", str(out_stack),
           "-find", "-points", str(out_model)]
    cmd += ["-scan", str(xr_cfg.pop("scan", 3.0)),
            "-xyscan", str(xr_cfg.pop("xyscan", 128)),
            "-edge", str(xr_cfg.pop("edge", 64))]
    cmd += _extra_args(xr_cfg)
    _run(cmd, dry_run, cfg)
    return out_stack


def bin_and_normalize(cfg: dict, stack: Path, out_dir: Path, dry_run: bool) -> tuple:
    """newstack -multadd (shift the minimum above 0, needed for SIRT) +
    -bin (to cfg.bin_factor) + -antialias. Returns (binned_stack_path,
    base_name_without_bin_suffix). header -minimum needs the real file to
    exist, so in dry-run mode a placeholder shift of 0 is used instead --
    it's never executed, just printed."""
    base = stack.stem.replace("_NoXray", "")
    bin_factor = cfg["bin_factor"]
    binned = out_dir / f"{base}_bin{bin_factor}.st"

    if dry_run:
        min_shift = 0
    else:
        result = subprocess.run([_bin(cfg, "header"), "-minimum", str(stack)],
                                 capture_output=True, text=True, check=True, env=_env(cfg))
        min_val = float(result.stdout.split()[0])
        min_shift = 10 - int(min_val)

    # newstack's usage is positional (`input_files... output_file`) --
    # confirmed from a real run: using -input/-output flags instead
    # triggers "ERROR: NEWSTACK - You must specify number of sections to
    # write to each output file" (those flags are for a different list-
    # accumulation mode, for combining many single-section files).
    # -mode 1, exactly as bin_tomo.sh uses.
    cmd = [_bin(cfg, "newstack"),
           "-multadd", f"1,{min_shift}",
           "-bin", str(bin_factor),
           "-antialias", "4",
           "-mode", "1",
           str(stack), str(binned)]
    _run(cmd, dry_run, cfg)
    return binned, base


def run_raptor(cfg: dict, binned_stack: Path, out_dir: Path, exec_path: str, dry_run: bool) -> dict:
    """RAPTOR fiducial tracking + alignment -- a direct translation of
    bin_tomo.sh's own RAPTOR call and the three mv commands right after
    it (.tlt, .xf, .ali promoted from RAPTOR's scratch output dir
    out_dir/<base>/ into flat out_dir/<base>.{tlt,xf,ali} files, scratch
    dir then deleted, same as bin_tomo.sh's RAP_PATH_OUT). -bin 1 because
    binned_stack is already pre-binned by bin_and_normalize() --
    RAPTOR's man page warns -diameter must be in the pixel size of
    whatever stack -input points at, which here is this already-binned
    stack (matches bin_tomo.sh's own proven math, see
    gold_diameter_px()).

    No retry, no output verification, no workaround for RAPTOR not
    writing a file -- if RAPTOR doesn't produce .tlt/.xf/.ali, the
    .rename() calls below fail with a plain FileNotFoundError, exactly
    like bin_tomo.sh's own `mv` would fail with "No such file or
    directory". If that happens, it's a RAPTOR/tiltalign problem on this
    install for this input, not something to paper over here."""
    base = binned_stack.stem
    raptor_scratch = out_dir / base
    rc_cfg = dict(cfg.get("raptor", {}))
    diameter = rc_cfg.pop("diameter_px", None) or gold_diameter_px(cfg)

    cmd = [_bin(cfg, "RAPTOR"),
           "-execPath", exec_path,
           "-path", str(binned_stack.parent),
           "-input", binned_stack.name,
           "-output", str(raptor_scratch),
           "-diameter", str(diameter),
           "-bin", "1"]
    if rc_cfg.pop("xray", False):
        cmd.append("-xray")
    if "markers_per_image" in rc_cfg and rc_cfg["markers_per_image"]:
        cmd += ["-markers", str(rc_cfg.pop("markers_per_image"))]
    else:
        rc_cfg.pop("markers_per_image", None)
    cmd += _extra_args(rc_cfg)
    _run(cmd, dry_run, cfg)

    tlt_dst, xf_dst, fid_dst, ali_dst = (out_dir / f"{base}.tlt", out_dir / f"{base}.xf",
                                          out_dir / f"{base}.fid.txt", out_dir / f"{base}.ali")
    if dry_run:
        return {"ali": ali_dst, "tlt": tlt_dst, "xf": xf_dst, "fid_txt": fid_dst, "base": base}

    (raptor_scratch / "IMOD" / f"{base}.tlt").rename(tlt_dst)
    (raptor_scratch / "IMOD" / f"{base}.xf").rename(xf_dst)
    (raptor_scratch / "IMOD" / f"{base}.fid.txt").rename(fid_dst)
    (raptor_scratch / "align" / f"{base}.ali").rename(ali_dst)

    shutil.rmtree(raptor_scratch, ignore_errors=True)
    return {"ali": ali_dst, "tlt": tlt_dst, "xf": xf_dst, "fid_txt": fid_dst, "base": base}


def erase_gold(cfg: dict, raptor_result: dict, out_dir: Path, dry_run: bool) -> Path:
    """Optional (imod.erase_gold.enabled): convert RAPTOR's text fiducial
    model (*.fid.txt) to a binary IMOD model with point2model -scat, then
    ccderaser -circle to erase each tracked bead position from the
    aligned stack -- the standard IMOD "erase gold" step, done here BEFORE
    reconstruction so the gold-erased version is what gets reconstructed.
    Returns the gold-erased .ali path."""
    base = raptor_result["base"]
    eg_cfg = dict(cfg.get("erase_gold", {}))
    eg_cfg.pop("enabled", None)
    fid_mod = out_dir / f"{base}_fid.mod"
    _run([_bin(cfg, "point2model"), "-scat",
          "-input", str(raptor_result["fid_txt"]),
          "-output", str(fid_mod)], dry_run, cfg)

    erased_ali = out_dir / f"{base}_noGold.ali"
    better_radius = eg_cfg.pop("better_radius_px", None)
    cmd = [_bin(cfg, "ccderaser"),
           "-input", str(raptor_result["ali"]),
           "-output", str(erased_ali),
           "-model", str(fid_mod),
           "-circle", "1"]
    if better_radius:
        cmd += ["-better", str(better_radius)]
    cmd += _extra_args(eg_cfg)
    _run(cmd, dry_run, cfg)
    return erased_ali


def reconstruct(cfg: dict, ali: Path, tlt: Path, out_dir: Path, suffix: str, dry_run: bool) -> Path:
    """tilt -RotateBy90 -THICKNESS ... -- WBP reconstruction (matches
    bin_tomo.sh's main .mrc output). suffix distinguishes this call's
    output from a later FakeSIRT/filtered reconstruction of the same
    aligned stack (e.g. "", "_FS10", "_flt")."""
    base = ali.stem
    out_mrc = out_dir / f"{base}{suffix}.mrc"
    rec_cfg = dict(cfg.get("reconstruction", {}))
    cmd = [_bin(cfg, "tilt"), "-RotateBy90",
           "-THICKNESS", str(cfg["thickness_px"]),
           "-tiltfile", str(tlt),
           "-InputProjections", str(ali),
           "-OutputFile", str(out_mrc)]
    fake_sirt = rec_cfg.pop("fake_sirt_iterations", None)
    if fake_sirt and suffix:
        cmd += ["-FakeSIRTiterations", str(fake_sirt)]
    cmd += _extra_args(rec_cfg)
    _run(cmd, dry_run, cfg)
    return out_mrc


def filtered_reconstruct(cfg: dict, ali: Path, tlt: Path, out_dir: Path, dry_run: bool) -> Path:
    """mtffilter (low/high-pass contrast filter) + tilt -- matches
    bin_tomo.sh's "_flt" filtered reconstruction. -low/-high in the
    original script are the real -lowpass/-highpass flags (confirmed via
    --help: -lowpass takes 2 floats [radius, sigma], -highpass takes 1
    [sigma])."""
    base = ali.stem
    flt_cfg = dict(cfg.get("filtered_reconstruction", {}))
    flt_cfg.pop("enabled", None)
    low_pass = flt_cfg.pop("low_pass", [0.2, 0.05])
    high_pass = flt_cfg.pop("high_pass", 0.0)
    flt_ali = out_dir / f"{base}_flt.ali"
    cmd = [_bin(cfg, "mtffilter"), "-input", str(ali), "-output", str(flt_ali),
           "-highpass", str(high_pass),
           "-lowpass", ",".join(str(v) for v in low_pass)]
    cmd += _extra_args(flt_cfg)
    _run(cmd, dry_run, cfg)
    return reconstruct(cfg, flt_ali, tlt, out_dir, suffix="", dry_run=dry_run)


def _extra_args(flags: dict) -> list:
    # IMOD's PIP-based argument parser (confirmed 2026-10-21 against a
    # real run + a synthetic MRC test): a multi-value option needs its
    # values comma-joined into ONE token ("-multadd 1,10"), not passed as
    # separate space-separated tokens ("-multadd 1 10") -- the latter
    # silently mis-parses the second value as the next positional
    # argument (a filename), e.g. "ERROR: iiOpen - Opening file 10".
    args = []
    for key, value in (flags or {}).items():
        args.append(f"-{key}")
        if isinstance(value, (list, tuple)):
            args.append(",".join(str(v) for v in value))
        elif value is not True:
            args.append(str(value))
    return args
