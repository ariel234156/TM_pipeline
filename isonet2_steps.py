#!/usr/bin/env python3
"""
Thin, generic wrappers around the isonet2 CLI (isonet.py).

Flag names/defaults for prepare_star, deconv, make_mask, refine and predict
were verified 2026-09-15 against `isonet.py <subcommand> --help` on a real
install (/home/isonet2/IsoNet2, pip-installed editable into the
isonet2_environment conda env) and matched almost exactly. `denoise` below
has NOT been separately verified the same way.

Every key in a pipeline YAML config's isonet stage dict (see
pipeline_config.example.yaml, "isonet:" section) is forwarded verbatim as
`--key value`, so any parameter IsoNet2 documents can be set without
touching this file.
"""
import os
import subprocess
from pathlib import Path

# isonet.py's refine/predict save preview PNGs via matplotlib. Without a
# real display, matplotlib can fall back to a Qt GUI backend (if PyQt5 is
# importable) that throws an "ICE default IO error handler" crash on
# process exit -- after all the real work (checkpoints/volumes) is already
# written to disk. Forcing a headless backend avoids that spurious failure.
_ISONET_ENV = dict(os.environ, MPLBACKEND="Agg", QT_QPA_PLATFORM="offscreen")


def _flag_args(flags: dict) -> list:
    args = []
    for key, value in flags.items():
        if value is None:
            continue
        args.append(f"--{key}")
        if isinstance(value, (list, tuple)):
            args.extend(str(v) for v in value)
        else:
            args.append(str(value))
    return args


def _run(cmd: list, dry_run: bool):
    print("+ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, env=_ISONET_ENV)


def prepare_star(even_dir, odd_dir, pixel_size, defocus, isonet_bin="isonet.py",
                  dry_run=False, **flags) -> Path:
    cmd = [isonet_bin, "prepare_star",
           "--even", str(even_dir), "--odd", str(odd_dir),
           "--pixel_size", str(pixel_size), "--defocus", str(defocus)]
    cmd += _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("star_name", "tomograms.star"))


def deconv(star_file, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    cmd = [isonet_bin, "deconv", str(star_file)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "./deconv"))


def make_mask(star_file, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    cmd = [isonet_bin, "make_mask", str(star_file)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "mask"))


def denoise(star_file, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    """Quick noise2noise pass -- useful for a fast preview / mask sanity
    check before committing to a full `refine` run. Not wired into
    run_pipeline.py's default stage order; call it manually or add it."""
    cmd = [isonet_bin, "denoise", str(star_file)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "denoise"))


def refine(star_file, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    cmd = [isonet_bin, "refine", str(star_file)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "isonet_maps"))


def predict(star_file, model, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    cmd = [isonet_bin, "predict", str(star_file), "--model", str(model)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "./corrected_tomos"))
