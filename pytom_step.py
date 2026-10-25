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

GOTCHA found 2026-10-19: IsoNet2's predict step does not always write the
real tomogram pixel size into its output MRC header -- pytom_match_
template.py falls back to reading voxel size from the MRC file if not
given explicitly, which can silently produce a massive scale mismatch.
This module never relies on the tomogram's own header: either
--voxel-size-angstrom is passed explicitly from pytom.voxel_size_angstrom,
or (preferred, see match_template_batch's docstring) --relion5-tomograms-
star is used instead, which derives voxel size (and tilt angles, dose,
real per-tilt CTF parameters) directly from a RELION5 tomograms.star
rather than the tomogram file at all.

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
    # pytom_create_template.py reads the input voxel size from the MRC
    # header if not given, and requires it to be <= the output voxel
    # size (it only downsamples). The reference map's header stores
    # 0.30528 A (float32 rounding noise from whatever tool wrote it)
    # instead of its real 0.305 A, so force the input voxel size
    # explicitly instead of trusting the header. input and output are
    # genuinely different values here: input is the reference map's own
    # native resolution, output is the tomogram's real pixel size (see
    # the GOTCHA comment on the pytom: config block).
    input_voxel_size = cfg.get("reference_map_voxel_size_angstrom", cfg["voxel_size_angstrom"])
    cmd = [_script(cfg, "pytom_create_template.py"),
           "-i", cfg["reference_map"],
           "-o", str(out_file),
           "--input-voxel-size-angstrom", str(input_voxel_size),
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
    extract_candidates step. Returns {tomogram_stem: destination_dir}.

    GOTCHA / real pytom feature (confirmed 2026-10-24 from pytom_tm's own
    source, entry_points.py + io.py's parse_relion5_star_data):
    pytom_match_template.py has a built-in --relion5-tomograms-star flag
    that auto-derives voxel size, tilt angles, dose accumulation, AND
    real per-tilt CTF parameters (defocus, voltage, Cs, amplitude
    contrast) straight from a RELION5 tomograms.star + its tilt_series/
    <name>.star -- "overwrite[s] all other metadata options" (its own
    --help text). We were NOT using this: our job.json always showed
    "ctf_data": null and "dose_accumulation": null, meaning matching ran
    with zero CTF/dose weighting. When cfg["relion5_tomograms_star"] is
    set, this now uses that flag instead of manually-typed --voxel-size-
    angstrom/-a, which also eliminates the hand-maintained voxel_size_
    angstrom config value as a source of scale-mismatch bugs entirely
    (see pipeline_config.yaml's pytom: GOTCHA comment on exactly that
    bug, found and fixed earlier the same way).

    CATCH: parse_relion5_star_data matches the tomogram by checking
    tomogram_filename.endswith(rlnTomoName) -- IsoNet2's own output
    names (e.g. "_isonet2_unet-medium_Position_1_3_final.mrc") don't
    satisfy this (they end in "_final", not the clean name). Worked
    around by symlinking each tomogram to its clean rlnTomoName under
    out_dir/tomograms_relion5/ (resumable -- skips existing links) and
    matching against those instead of the raw tomogram_dir files."""
    mt_cfg = dict(cfg.get("match_template", {}))
    gpu = [str(g) for g in cfg.get("gpu", [0])]
    tilt_angles = mt_cfg.pop("tilt_angles", None)
    relion5_star = cfg.get("relion5_tomograms_star")
    destinations = {}

    mrc_files = sorted(Path(tomogram_dir).glob("*.mrc"))
    if not mrc_files:
        raise RuntimeError(f"no .mrc tomograms found in {tomogram_dir}")

    if relion5_star and not dry_run:
        clean_names = _read_clean_tomo_names(Path(relion5_star))
        symlink_dir = Path(cfg["out_dir"]) / "tomograms_relion5"
        symlink_dir.mkdir(parents=True, exist_ok=True)
        renamed = []
        for tomo in mrc_files:
            matches = [c for c in clean_names if c in tomo.stem]
            if not matches:
                raise RuntimeError(f"no clean rlnTomoName found inside {tomo.stem!r} "
                                    f"(from {tomo}) -- checked against {clean_names}")
            link = symlink_dir / f"{max(matches, key=len)}.mrc"
            if not link.exists():
                link.symlink_to(tomo.resolve())
            renamed.append(link)
        mrc_files = renamed

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
               "-g"] + gpu
        if relion5_star:
            cmd += ["--relion5-tomograms-star", str(relion5_star)]
        else:
            cmd += ["--voxel-size-angstrom", str(cfg["voxel_size_angstrom"])]
            if tilt_angles is not None:
                cmd += ["-a"] + [str(a) for a in tilt_angles]
        cmd += _extra_args(mt_cfg)
        _run(cmd, dry_run)

    return destinations


def extract_candidates_batch(cfg: dict, destinations: dict, dry_run: bool) -> list:
    """Run pytom_extract_candidates.py once per tomogram's job.json
    (written by match_template_batch into each destination dir). Returns
    the list of output .star file paths.

    GOTCHA (confirmed 2026-10-22 against RELION5 5.0.1's own source,
    src/jaz/tomography/particle_set.cpp): pytom's rlnTomoName comes from
    the tomogram file's own stem (e.g. IsoNet2 predict output
    "_isonet2_unet-medium_Position_1_3_final"), which does NOT match the
    clean tilt series name in relion5_star's tomograms.star
    ("Position_1_3"). RELION5 auto-builds its optics table by exact
    rlnTomoName match against tomograms.star when particles.star has no
    optics block of its own -- a mismatch here means the auto-build
    silently fails to link any particle to a real tomogram, and you get
    "rlnTomoTiltSeriesPixelSize missing from optics MetaDataTable" with
    no mention of the real cause. Fixed by remapping rlnTomoName to the
    clean name (see relion5_tomograms_star config key) right after
    extraction, before merge_stars concatenates everything."""
    ec_cfg = dict(cfg.get("extract_candidates", {}))
    n_particles = ec_cfg.pop("number_of_particles", 500)
    star_files = []
    clean_names = (_read_clean_tomo_names(Path(cfg["relion5_tomograms_star"]))
                   if cfg.get("relion5_tomograms_star") and not dry_run else None)

    for stem, dest in destinations.items():
        job_file = dest / "scores.json" if dry_run else _find_job_file(dest)
        cmd = [_script(cfg, "pytom_extract_candidates.py"),
               "-j", str(job_file),
               "-n", str(n_particles),
               "--particle-diameter", str(cfg["particle_diameter_angstrom"])]
        cmd += _extra_args(ec_cfg)
        _run(cmd, dry_run)
        star_file = dest / f"{stem}_particles.star"
        if clean_names is not None:
            _remap_tomo_name(star_file, clean_names)
        star_files.append(star_file)

    return star_files


def _read_clean_tomo_names(tomograms_star: Path) -> list:
    import starfile
    df = starfile.read(tomograms_star)
    if isinstance(df, dict):
        df = next(iter(df.values()))
    return df["rlnTomoName"].astype(str).tolist()


def _remap_tomo_name(star_file: Path, clean_names: list) -> None:
    """Rewrite rlnTomoName in-place to whichever clean_names entry is the
    longest substring match of the raw value (longest wins so e.g.
    "Position_1" can't shadow "Position_1_3")."""
    import starfile
    df = starfile.read(star_file)
    rename = {}
    for raw in df["rlnTomoName"].astype(str).unique():
        matches = [c for c in clean_names if c in raw]
        if not matches:
            raise RuntimeError(f"no clean tomogram name found inside {raw!r} "
                                f"(from {star_file}) -- checked against {clean_names}")
        rename[raw] = max(matches, key=len)
    df["rlnTomoName"] = df["rlnTomoName"].astype(str).replace(rename)
    # must keep the "particles" block name explicit -- starfile.write on a
    # bare DataFrame (no block name) writes an unnamed "data_" block instead
    # of "data_particles", which merge_stars's input files are expected to have
    starfile.write({"particles": df}, star_file, overwrite=True)


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
