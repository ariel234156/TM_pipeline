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


def _run(cmd: list, dry_run: bool, cwd=None):
    print("+ " + " ".join(cmd))
    if not dry_run:
        if cwd is not None:
            Path(cwd).mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True, env=_ISONET_ENV, cwd=cwd)


def _absolute(value):
    """A path argument made absolute against the current directory, so it
    means the same after the command is started in another one. IsoNet2's
    "None" placeholder is left alone."""
    return value if value in (None, "None") else os.path.abspath(str(value))


def _isonet_prefix(isonet_bin) -> list:
    """isonet_bin can be a plain command/path string (resolved via PATH at
    run time -- depends on the right conda env being activated first), or
    a [python_path, script_path] pair that pins the exact interpreter
    regardless of shell activation state. The latter is what avoids a
    prepare_star/refine/etc. silently picking up a *different* isonet.py
    (e.g. a stale one from another install left on PATH) with the wrong
    dependencies -- see pipeline_config.yaml's isonet_bin comment."""
    return list(isonet_bin) if isinstance(isonet_bin, (list, tuple)) else [isonet_bin]


def prepare_star(pixel_size, defocus=None, even_dir=None, odd_dir=None, full_dir=None,
                  isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    """Pass either (even_dir and odd_dir) for the even/odd half-tomogram
    workflow (isonet2-n2n), or full_dir for the single-full-tomogram
    workflow (isonet2) -- e.g. AreTomo2 output, which has no even/odd
    split. Exactly one of the two must be given. defocus is optional --
    omit it (leave None) to let isonet.py fall back to its own default
    (e.g. when no per-tomogram CTF info is available, as with AreTomo2's
    output -- see aretomo2_step.py).

    isonet.py prepare_star also writes a .to_node.json (a copy of the star
    table, for its GUI) into the directory it is started from. Started
    from wherever ./pipeline was run -- usually the pipeline's own folder,
    which other accounts can't write -- it failed for them after writing
    the star file. So it runs in the star file's folder, with every path
    argument made absolute first."""
    if full_dir is not None:
        cmd = _isonet_prefix(isonet_bin) + ["prepare_star", "--full", _absolute(full_dir)]
    elif even_dir is not None and odd_dir is not None:
        cmd = _isonet_prefix(isonet_bin) + [
            "prepare_star", "--even", _absolute(even_dir), "--odd", _absolute(odd_dir)]
    else:
        raise ValueError("prepare_star needs either full_dir, or both even_dir and odd_dir")
    flags = dict(flags)
    for key in ("star_name", "mask_folder", "coordinate_folder"):
        if key in flags:
            flags[key] = _absolute(flags[key])
    star = Path(flags.setdefault("star_name", _absolute("tomograms.star")))
    cmd += ["--pixel_size", str(pixel_size)]
    if defocus is not None:
        cmd += ["--defocus", str(defocus)]
    cmd += _flag_args(flags)
    _run(cmd, dry_run, cwd=star.parent)
    return star


def deconv(star_file, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    cmd = _isonet_prefix(isonet_bin) + ["deconv", str(star_file)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "./deconv"))


def make_mask(star_file, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    cmd = _isonet_prefix(isonet_bin) + ["make_mask", str(star_file)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "mask"))


def denoise(star_file, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    """Quick noise2noise pass -- useful for a fast preview / mask sanity
    check before committing to a full `refine` run. Not wired into
    run_pipeline.py's default stage order; call it manually or add it."""
    cmd = _isonet_prefix(isonet_bin) + ["denoise", str(star_file)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "denoise"))


def refine(star_file, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    cmd = _isonet_prefix(isonet_bin) + ["refine", str(star_file)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "isonet_maps"))


def predict(star_file, model, isonet_bin="isonet.py", dry_run=False, **flags) -> Path:
    cmd = _isonet_prefix(isonet_bin) + ["predict", str(star_file), "--model", str(model)] + _flag_args(flags)
    _run(cmd, dry_run)
    return Path(flags.get("output_dir", "./corrected_tomos"))
