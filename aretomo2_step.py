#!/usr/bin/env python3
"""
Batch-run AreTomo2 (alignment + reconstruction only, no motion correction)
over a directory of pre-assembled tilt series MRC stacks -- one call per
.mrc file. Unlike AreTomo3, AreTomo2 has no built-in directory-batch mode;
this mirrors the shell-loop pattern AreTomo2 is normally run with.

Flag names/defaults verified 2026-10-19 against a real install
(/home/AreTomo2/AreTomo2, version 1.1.2) via --help AND empirically --
two entries in --help ("$-10s" / "-10s") are garbled by a printf bug in
this build, confirmed by testing that they really are -AmpContrast and
-ExtPhase (matching AreTomo3's flags of the same name/behavior). Note
this build's real compiled defaults for -AlignZ (600) and -VolZ (1200)
do NOT match what --help claims (256 and 0 respectively) -- same kind of
doc/binary mismatch already found in AreTomo3. -Patch (not -AtPatch/
-McPatch) is AreTomo2's single local-alignment patch-grid flag, since it
has no separate motion-correction stage to apply McPatch to.

AreTomo2 produces no CTF.txt-equivalent output file (confirmed against a
real prior run's output directory) -- use organize.defocus_source: none
for prepare_star when feeding this stage's output onward, since there is
no automatic per-tomogram defocus to extract. Pass -PixSize/-Kv/-Cs/
-ImgDose to enable AreTomo2's own internal CTF-aware dose weighting if
wanted; this is independent of prepare_star's per-tomogram --defocus.

Every tunable AreTomo2 parameter lives in the YAML config (pipeline_config
.yaml's "aretomo2:" section) -- edit the YAML, not this file.
"""
import subprocess
from pathlib import Path

FLAG_MAP = {
    "aln_file": "-AlnFile",
    "ang_file": "-AngFile",
    "tmp_file": "-TmpFile",
    "log_file": "-LogFile",
    "tilt_range": "-TiltRange",
    "tilt_axis": "-TiltAxis",
    "align_z": "-AlignZ",
    "vol_z": "-VolZ",
    "out_bin": "-OutBin",
    "gpu": "-Gpu",
    "tilt_cor": "-TiltCor",
    "recon_range": "-ReconRange",
    "pix_size": "-PixSize",
    "kv": "-Kv",
    "img_dose": "-ImgDose",
    "cs": "-Cs",
    "amp_contrast": "-AmpContrast",
    "ext_phase": "-ExtPhase",
    "flip_vol": "-FlipVol",
    "flip_int": "-FlipInt",
    "sart": "-Sart",
    "wbp": "-Wbp",
    "patch": "-Patch",
    "dark_tol": "-DarkTol",
    "tilt_scheme": "-TiltScheme",
    "out_xf": "-OutXF",
    "out_imod": "-OutImod",
    "align": "-Align",
    "crop_vol": "-CropVol",
    "bft": "-Bft",
    "intp_cor": "-IntpCor",
}


def _build_shared_flags(cfg: dict) -> list:
    cmd = []
    for key, flag in FLAG_MAP.items():
        if key not in cfg or cfg[key] is None:
            continue
        value = cfg[key]
        cmd.append(flag)
        if isinstance(value, (list, tuple)):
            cmd.extend(str(v) for v in value)
        else:
            cmd.append(str(value))

    for flag, value in (cfg.get("extra") or {}).items():
        cmd.append(flag)
        if isinstance(value, (list, tuple)):
            cmd.extend(str(v) for v in value)
        else:
            cmd.append(str(value))

    return cmd


def run_aretomo2_batch(cfg: dict, binary: str = "AreTomo2", dry_run: bool = False) -> Path:
    """Run AreTomo2 once per *.mrc file in cfg['in_dir'], writing each
    result to cfg['out_dir']. Output filenames are <stem><out_suffix>.mrc,
    matching the naming convention of the aretomo.sh-style batch scripts
    this mirrors. skip_existing (default True) skips any input file whose
    output .mrc already exists, since AreTomo2 itself has no equivalent of
    AreTomo3's -Resume/MdocDone.txt tracking."""
    in_dir = Path(cfg["in_dir"])
    out_dir = Path(cfg["out_dir"])
    suffix = cfg.get("out_suffix", "_AreTomo")
    skip_existing = cfg.get("skip_existing", True)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_flags = _build_shared_flags(cfg)
    mrc_files = sorted(in_dir.glob("*.mrc"))
    if not mrc_files:
        raise RuntimeError(f"no .mrc files found in {in_dir}")

    for mrc in mrc_files:
        out_mrc = out_dir / f"{mrc.stem}{suffix}.mrc"
        if skip_existing and out_mrc.exists() and not dry_run:
            print(f"+ [skip] {out_mrc} already exists")
            continue
        cmd = [binary, "-InMrc", str(mrc), "-OutMrc", str(out_mrc)] + shared_flags
        print("+ " + " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)

    return out_dir
