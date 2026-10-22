#!/usr/bin/env python3
"""
Template matching via PyTom's pytom-match-pick toolkit (pytom_tm conda env):
create_template -> create_mask -> match_template (per tomogram) ->
extract_candidates (per tomogram) -> merge_stars.

Flags verified 2026-10-19 against a real install (conda env pytom_tm) via
each script's own --help. The scripts' shebangs are absolute
(#!/home/coari/.conda/envs/pytom_tm/bin/python), so invoking them by full
path is immune to PATH/conda-activation state entirely -- no interpreter
pinning needed, unlike isonet.py (see isonet2_steps.py's _isonet_prefix
docstring for that story). bin_dir in the config is just the directory
those scripts live in.

GOTCHA found 2026-10-19: IsoNet2's predict step does not write the real
tomogram pixel size into its output MRC header (observed: header says
1.0 A/px regardless of the actual value) -- pytom_match_template.py falls
back to reading voxel size from the MRC file if not given explicitly,
which would silently produce a massive scale mismatch. This module always
passes --voxel-size-angstrom explicitly from the config; never rely on
the tomogram's own header for this.

create_mask's box size and radius (in pixels) are computed automatically
from create_template's actual output box size and the configured
particle_diameter_angstrom / voxel_size_angstrom, rather than requiring
the user to precompute and keep them in sync by hand.
"""
import subprocess
from pathlib import Path


def _run(cmd: list, dry_run: bool):
    print("+ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _script(cfg: dict, name: str) -> str:
    return str(Path(cfg["bin_dir"]) / name)


def _extra_args(flags: dict) -> list:
    args = []
    for key, value in (flags or {}).items():
        args.append(f"--{key}")
        if isinstance(value, (list, tuple)):
            args.extend(str(v) for v in value)
        elif value is not True:
            args.append(str(value))
    return args


def create_template(cfg: dict, dry_run: bool) -> Path:
    out_file = Path(cfg["out_dir"]) / "template.mrc"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_script(cfg, "pytom_create_template.py"),
           "-i", cfg["reference_map"],
           "-o", str(out_file),
           # pytom_create_template.py reads the input voxel size from the
           # MRC header if not given, and requires it to be <= the output
           # voxel size (it only downsamples). The reference map's header
           # stores 0.30528 A (float32 rounding noise from whatever tool
           # wrote it) instead of the intended 0.305 A, which trips that
           # check even though the two are physically the same spacing --
           # so force the input voxel size to the real, intended value.
           "--input-voxel-size-angstrom", str(cfg["voxel_size_angstrom"]),
           "--output-voxel-size-angstrom", str(cfg["voxel_size_angstrom"])]
    if cfg.get("center", True):
        cmd.append("--center")
    cmd += _extra_args(cfg.get("create_template", {}))
    _run(cmd, dry_run)
    return out_file


def create_mask(cfg: dict, template_file: Path, dry_run: bool) -> Path:
    out_file = Path(cfg["out_dir"]) / "mask.mrc"
    if dry_run and not template_file.exists():
        box_size = cfg.get("_assumed_box_size", 64)
    else:
        import mrcfile  # only needed here -- keeps pytom_step.py importable without it
        with mrcfile.open(template_file, permissive=True) as f:
            box_size = f.data.shape[0]
    radius_px = (cfg["particle_diameter_angstrom"] / 2.0) / cfg["voxel_size_angstrom"]
    mc_cfg = dict(cfg.get("create_mask", {}))
    sigma = mc_cfg.pop("sigma", 1.0)
    cmd = [_script(cfg, "pytom_create_mask.py"),
           "-b", str(box_size),
           "-o", str(out_file),
           "-r", str(round(radius_px, 2)),
           "-s", str(sigma),
           "--voxel-size", str(cfg["voxel_size_angstrom"])]
    cmd += _extra_args(mc_cfg)
    _run(cmd, dry_run)
    return out_file


def match_template_batch(cfg: dict, template_file: Path, mask_file: Path,
                          tomogram_dir: Path, dry_run: bool) -> dict:
    """Run pytom_match_template.py once per *.mrc in tomogram_dir, each
    into its own subfolder of out_dir/match (named after the tomogram's
    stem), so each tomogram's job.json stays unambiguous for the
    extract_candidates step. Returns {tomogram_stem: destination_dir}."""
    mt_cfg = dict(cfg.get("match_template", {}))
    gpu = [str(g) for g in cfg.get("gpu", [0])]
    tilt_angles = mt_cfg.pop("tilt_angles", None)
    destinations = {}

    mrc_files = sorted(Path(tomogram_dir).glob("*.mrc"))
    if not mrc_files:
        raise RuntimeError(f"no .mrc tomograms found in {tomogram_dir}")

    for tomo in mrc_files:
        dest = Path(cfg["out_dir"]) / "match" / tomo.stem
        dest.mkdir(parents=True, exist_ok=True)
        destinations[tomo.stem] = dest
        cmd = [_script(cfg, "pytom_match_template.py"),
               "-t", str(template_file),
               "-v", str(tomo),
               "-d", str(dest),
               "-m", str(mask_file),
               "--particle-diameter", str(cfg["particle_diameter_angstrom"]),
               "--voxel-size-angstrom", str(cfg["voxel_size_angstrom"]),
               "-g"] + gpu
        if tilt_angles is not None:
            cmd += ["-a"] + [str(a) for a in tilt_angles]
        cmd += _extra_args(mt_cfg)
        _run(cmd, dry_run)

    return destinations


def extract_candidates_batch(cfg: dict, destinations: dict, dry_run: bool) -> list:
    """Run pytom_extract_candidates.py once per tomogram's job.json
    (written by match_template_batch into each destination dir). Returns
    the list of output .star file paths."""
    ec_cfg = dict(cfg.get("extract_candidates", {}))
    n_particles = ec_cfg.pop("number_of_particles", 500)
    star_files = []

    for stem, dest in destinations.items():
        job_file = dest / "scores.json" if dry_run else _find_job_file(dest)
        cmd = [_script(cfg, "pytom_extract_candidates.py"),
               "-j", str(job_file),
               "-n", str(n_particles),
               "--particle-diameter", str(cfg["particle_diameter_angstrom"])]
        cmd += _extra_args(ec_cfg)
        _run(cmd, dry_run)
        star_files.append(dest / f"{stem}_particles.star")

    return star_files


def _find_job_file(dest: Path) -> Path:
    candidates = sorted(dest.glob("*_job.json"))
    if not candidates:
        raise RuntimeError(
            f"no *_job.json found in {dest} -- pytom_match_template.py should have "
            f"written one; check that stage completed successfully")
    return candidates[0]


def merge_stars(cfg: dict, dry_run: bool) -> Path:
    match_dir = Path(cfg["out_dir"]) / "match"
    out_file = Path(cfg["out_dir"]) / cfg.get("merged_star_name", "all_particles.star")
    # pytom_merge_stars.py merges every *.star directly inside -i, but our
    # per-tomogram star files each live in their own match/<stem>/ subfolder
    # -- symlink them into one flat directory first.
    merge_input_dir = Path(cfg["out_dir"]) / "merge_input"
    merge_input_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        import os
        for star in sorted(match_dir.glob("*/*_particles.star")):
            link = merge_input_dir / f"{star.parent.name}_{star.name}"
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(star.resolve(), link)

    cmd = [_script(cfg, "pytom_merge_stars.py"),
           "-i", str(merge_input_dir),
           "-o", str(out_file)]
    _run(cmd, dry_run)
    return out_file
