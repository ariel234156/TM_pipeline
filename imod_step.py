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

CAVEAT (acknowledged, not re-tested this session): RAPTOR needs real gold
fiducial markers in the sample. No fiducial-containing dataset was
available to test the real RAPTOR run end-to-end this session -- every
flag above was verified against this install's --help/man text and the
previously-working bin_tomo.sh script's own (empirically proven) output
paths, but the actual RAPTOR invocation itself has NOT been re-run here.
Validate the first real run carefully.
"""
import math
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list, dry_run: bool):
    print("+ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


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
    _run([_bin(cfg, "extracttilts"), "-mdoc", str(mdoc_path), str(angles_path)], dry_run)

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
    _run(cmd, dry_run)
    return stack_path, base


def extract_rawtlt(cfg: dict, stack: Path, out_name: Path, dry_run: bool) -> Path:
    """extracttilts (header mode): pulls tilt angle back out of the
    stack's own section headers. Matches bin_tomo.sh's first step exactly
    (it names the output using the FUTURE binned stack's basename even
    though extraction runs on the pre-binned file -- tilt angle doesn't
    change with binning, only the filename needs to match later)."""
    _run([_bin(cfg, "extracttilts"), str(stack), str(out_name)], dry_run)
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
    _run(cmd, dry_run)
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
                                 capture_output=True, text=True, check=True)
        min_val = float(result.stdout.split()[0])
        min_shift = 10 - int(min_val)

    # newstack's usage is positional (`input_files... output_file`) --
    # confirmed from a real run: using -input/-output flags instead
    # triggers "ERROR: NEWSTACK - You must specify number of sections to
    # write to each output file" (those flags are for a different list-
    # accumulation mode, for combining many single-section files).
    #
    # -mode 2 (float), NOT -mode 1 (16-bit int) -- confirmed against a
    # real fiducial dataset: RAPTOR's own MRC reader fails on mode 1 with
    # a generic "Error opening file ... not enough memory" (misleading --
    # the file is fine, RAPTOR just can't read that mode), while mode 2
    # let it track fiducials successfully (58/61 markers found).
    cmd = [_bin(cfg, "newstack"),
           "-multadd", f"1,{min_shift}",
           "-bin", str(bin_factor),
           "-antialias", "4",
           "-mode", "2",
           str(stack), str(binned)]
    _run(cmd, dry_run)
    return binned, base


def _section_count(cfg: dict, stack: Path) -> int:
    result = subprocess.run([_bin(cfg, "header"), "-size", str(stack)],
                             capture_output=True, text=True, check=True)
    return int(result.stdout.split()[2])


def run_raptor(cfg: dict, binned_stack: Path, out_dir: Path, exec_path: str, dry_run: bool) -> dict:
    """RAPTOR fiducial tracking + alignment. -bin 1 because binned_stack
    is already pre-binned by bin_and_normalize() -- RAPTOR's own man page
    warns the -diameter value must be in the pixel size of whatever stack
    -input actually points at, which here is this already-binned stack
    (matches bin_tomo.sh's own proven math, see gold_diameter_px()).

    Path structure matches bin_tomo.sh exactly: RAPTOR writes into a
    scratch subfolder named just after the input stack (out_dir/<base>/,
    same as bin_tomo.sh's RAP_PATH_OUT=${CURRENT_PATH}/${NAME_IN}), then
    the essential files (.tlt/.xf/.fid.txt) get promoted to flat
    <base>.tlt/.xf/.fid.txt files directly in out_dir and the scratch
    subfolder is deleted -- same mv+rm pattern bin_tomo.sh used.

    RAPTOR's own exit code is NOT reliable (confirmed empirically: it can
    report 0 even when an internal step segfaults), so success is judged
    by whether .tlt/.xf/.fid.txt actually exist afterward, not by the
    return code.

    GOTCHA (confirmed empirically, 2026-10-21, on a real fiducial
    dataset): this RAPTOR build does NOT write the aligned stack (*.ali)
    itself by default, despite its own man page saying verbosity 1
    (default) keeps one -- and passing -rec/-thickness to have RAPTOR
    generate it internally instead SEGFAULTS (confirmed with both -rec 0
    and -rec 2, same crash point, exit code still reported as 0). So the
    .ali is built here instead, from the pre-binned stack + RAPTOR's own
    .xf, via build_aligned_stack() -- see its docstring for the real
    reason this needs an extra step (RAPTOR's .xf can have fewer lines
    than the input has sections, when some views get dropped during
    robust fitting)."""
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
    _run(cmd, dry_run)

    tlt_dst, xf_dst, fid_dst, ali_dst = (out_dir / f"{base}.tlt", out_dir / f"{base}.xf",
                                          out_dir / f"{base}.fid.txt", out_dir / f"{base}.ali")
    if dry_run:
        return {"ali": ali_dst, "tlt": tlt_dst, "xf": xf_dst, "fid_txt": fid_dst, "base": base}

    for name, dst in [("tlt", tlt_dst), ("xf", xf_dst), ("fid.txt", fid_dst)]:
        src = raptor_scratch / "IMOD" / f"{base}.{name}"
        if not src.exists():
            raise RuntimeError(
                f"RAPTOR did not produce {src} (its exit code is not reliable -- check "
                f"{raptor_scratch}/align/{base}_RAPTOR.log for what actually happened)")
        src.rename(dst)

    ali_src = raptor_scratch / "align" / f"{base}.ali"
    if ali_src.exists():
        ali_src.rename(ali_dst)
    else:
        build_aligned_stack(cfg, binned_stack, xf_dst, ali_dst, dry_run)

    shutil.rmtree(raptor_scratch, ignore_errors=True)
    return {"ali": ali_dst, "tlt": tlt_dst, "xf": xf_dst, "fid_txt": fid_dst, "base": base}


def build_aligned_stack(cfg: dict, binned_stack: Path, xf: Path, out_ali: Path, dry_run: bool) -> Path:
    """Build the aligned stack ourselves when RAPTOR didn't write one
    (see run_raptor()'s docstring). newstack -xform needs exactly one
    transform line per INPUT section -- but this install's tiltalign can
    crash partway through WRITING .xf, silently truncating it mid-line
    (CONFIRMED on a real dataset: the raw file ended mid-number at
    exactly a 4096-byte buffer boundary, with no trailing newline -- a
    genuine bug in this tiltalign build, not a real "excluded views"
    list from robust fitting as first suspected). newstack itself
    reports the exact usable line count when this happens ("ERROR:
    NEWSTACK - TRANSFORM LINE number out of bounds: N"), so try applying
    -xform directly first and only fall back to trimming if that error
    appears. Since the write always proceeds in view order from view 0,
    a truncated write is always missing views from the END, never the
    start -- confirmed empirically: excluding the FIRST n_drop sections
    still failed this same bounds check, excluding the LAST n_drop
    succeeded. Check the resulting .ali in 3dmod regardless -- this is a
    workaround for what looks like a real bug in this specific tiltalign
    build, not documented, guaranteed-stable behavior."""
    if dry_run:
        out_ali.parent.mkdir(parents=True, exist_ok=True)
        _run([_bin(cfg, "newstack"), "-xform", str(xf), str(binned_stack), str(out_ali)], dry_run)
        return out_ali

    cmd = [_bin(cfg, "newstack"), "-xform", str(xf), str(binned_stack), str(out_ali)]
    print("+ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return out_ali

    n_xf = None
    for line in (result.stdout + result.stderr).splitlines():
        if "out of bounds" in line:
            n_xf = int(line.split()[-1])
    if n_xf is None:
        raise RuntimeError(f"newstack -xform failed for a reason other than a truncated .xf "
                            f"(no 'out of bounds' in its output): {result.stderr}")

    n_sections = _section_count(cfg, binned_stack)
    n_drop = n_sections - n_xf
    print(f"build_aligned_stack: {xf.name} has only {n_xf} usable transform lines but "
          f"{binned_stack.name} has {n_sections} sections (likely a truncated .xf write -- "
          f"see build_aligned_stack's docstring) -- excluding the last {n_drop} sections")
    trimmed = out_ali.with_name(f"{binned_stack.stem}_trimmed.st")
    exclude_list = ",".join(str(n_sections - 1 - i) for i in range(n_drop))
    _run([_bin(cfg, "newstack"), "-exclude", exclude_list, str(binned_stack), str(trimmed)], dry_run)
    _run([_bin(cfg, "newstack"), "-xform", str(xf), str(trimmed), str(out_ali)], dry_run)
    trimmed.unlink(missing_ok=True)
    return out_ali


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
          "-output", str(fid_mod)], dry_run)

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
    _run(cmd, dry_run)
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
    _run(cmd, dry_run)
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
    _run(cmd, dry_run)
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
