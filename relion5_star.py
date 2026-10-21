#!/usr/bin/env python3
"""
Generate a RELION5-compatible tomograms.star (+ one per-tomogram
tilt_series/<name>.star) from either AreTomo engine's output.

Schema verified 2026-10-19 two ways:
  1. Against PyTom's own RELION5 reader (pytom_tm.io.parse_relion5_star_data)
     by reading its source directly -- this is what needs
     rlnTomoTiltSeriesPixelSize, rlnTomoTomogramBinning, rlnTomoHand,
     rlnDefocusV/rlnDefocusU on the per-tilt-series file.
  2. Against a REAL `relion_tomo_import SerialEM` run on this project's
     own .mdoc/.eer files (RELION5 5.0, installed in this environment) --
     this is RELION5's own authentic import-stage output, and is what
     revealed two things the first version of this module got wrong/
     missing (for the engine: aretomo3 path):

     a) BUG FIXED: rlnMicrographPreExposure is cumulative dose INCLUDING
        the current image (ends at this exposure), not strictly-before
        (starting at 0 for the first image) as this module originally
        computed. Confirmed from real output: the first-acquired tilt's
        PreExposure equals its own ExposureDose, not 0.
     b) Added rlnMicrographMovieName (from the .mdoc's SubFramePath,
        basename only), rlnTomoTiltMovieFrameCount (.mdoc NumSubFrames),
        rlnTomoNominalTiltAxisAngle (.mdoc TiltAxisAngle, repeated per
        row -- this is RELION5's own convention, confirmed from real
        output), rlnTomoNominalDefocus (.mdoc TargetDefocus -- the
        per-session target, NOT the per-image measured value),
        rlnMicrographOriginalPixelSize and rlnOpticsGroupName on the
        top-level file.
     Verified: output for a real tilt series matched the genuine RELION5
     import's values exactly (pre-exposure, tilt axis angle, nominal
     defocus, row count all identical).

GOTCHA (engine: aretomo3 path, confirmed empirically both from reading
AreTomo3's own files and matching real RELION5 import row order):
AreTomo3's own _TLT.txt/_CTF.txt are sorted by TILT ANGLE, not
acquisition order -- dose-symmetric collection schemes acquire
non-monotonically (e.g. 0, 2, -2, -4, 4, 6, -6, ...). Rows are written in
true acquisition order (iterating the .mdoc's [ZValue] sections), pulling
per-image measured defocus (rlnDefocusV/rlnDefocusU) from AreTomo3's
angle-sorted _CTF.txt by matching tilt angle.

engine: aretomo2 (AreTomo2) path: AreTomo2 takes an already-assembled
tilt series MRC per position and does no CTF estimation of its own, so
there is no measured per-image defocus to draw on (unlike engine 3's
AreTomo3 _CTF.txt). Instead this uses the companion .mdoc file's own
per-image "Defocus" field (SerialEM's own on-the-fly estimate -- less
refined than AreTomo3's actual CTF fit) for both rlnDefocusV and
rlnDefocusU, and a configurable constant for rlnTomoHand (no per-dataset
measurement available at all for this path; RELION5 itself defaults this
to 1 when not otherwise known). Requires a companion <base>.mdoc file
next to each AreTomo2 input .mrc -- if your pre-assembled stacks don't
have one (e.g. produced by a tool that doesn't preserve it), this path
has no tilt-angle source to build from at all and will skip that tomogram.

This has NOT been validated by actually running the resulting
tomograms.star through relion_tomo_align_tilt_series or any further
RELION5 job -- only against relion_tomo_import's raw output schema and
PyTom's reader.
"""
from pathlib import Path

import organize_for_isonet2 as prep

ANGLE_MATCH_TOLERANCE = 0.1  # degrees

TOMOGRAMS_HEADER = [
    "_rlnTomoName", "_rlnTomoTiltSeriesStarFile", "_rlnVoltage",
    "_rlnSphericalAberration", "_rlnAmplitudeContrast",
    "_rlnMicrographOriginalPixelSize", "_rlnTomoTiltSeriesPixelSize",
    "_rlnTomoTomogramBinning", "_rlnTomoHand", "_rlnOpticsGroupName",
]
TILT_SERIES_HEADER = [
    "_rlnMicrographMovieName", "_rlnTomoTiltMovieFrameCount",
    "_rlnTomoNominalStageTiltAngle", "_rlnTomoNominalTiltAxisAngle",
    "_rlnMicrographPreExposure", "_rlnTomoNominalDefocus",
    "_rlnDefocusV", "_rlnDefocusU",
]


def parse_mdoc(mdoc_path: Path) -> tuple:
    """Return (tilt_axis_angle, [per-image dicts in acquisition order]).
    Each per-image dict has: tilt_angle, exposure_dose, movie_name,
    frame_count, target_defocus, defocus (SerialEM's own per-image
    estimate, distinct from target_defocus -- see module docstring)."""
    tilt_axis_angle = None
    images = []
    current = {}
    with open(mdoc_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("[T =") and "TiltAxisAngle" in line:
                # e.g. "[T =   TiltAxisAngle = -92.67  Binning = 1  SpotSize = 6]"
                part = line.split("TiltAxisAngle", 1)[1].split("=", 1)[1]
                tilt_axis_angle = float(part.strip().split()[0])
            elif line.startswith("[ZValue"):
                if current:
                    images.append(current)
                current = {}
            elif line.startswith("TiltAngle"):
                current["tilt_angle"] = float(line.split("=", 1)[1].strip())
            elif line.startswith("ExposureDose"):
                current["exposure_dose"] = float(line.split("=", 1)[1].strip())
            elif line.startswith("TargetDefocus"):
                current["target_defocus"] = float(line.split("=", 1)[1].strip())
            elif line.startswith("Defocus"):
                current["defocus"] = float(line.split("=", 1)[1].strip())
            elif line.startswith("NumSubFrames"):
                current["frame_count"] = int(float(line.split("=", 1)[1].strip()))
            elif line.startswith("SubFramePath"):
                raw = line.split("=", 1)[1].strip()
                current["movie_name"] = raw.replace("\\", "/").rsplit("/", 1)[-1]
        if current:
            images.append(current)

    if tilt_axis_angle is None:
        raise ValueError(f"no TiltAxisAngle found in {mdoc_path}")
    return tilt_axis_angle, images


def compute_pre_exposure_by_angle(images: list) -> dict:
    """Cumulative dose INCLUDING each image's own exposure (RELION5's own
    convention, confirmed against a real relion_tomo_import run -- NOT
    strictly-before), keyed by rounded tilt angle for tolerant matching."""
    pre_exposure = {}
    cumulative = 0.0
    for image in images:
        cumulative += image["exposure_dose"]
        pre_exposure[round(image["tilt_angle"], 1)] = cumulative
    return pre_exposure


def read_ctf_full(ctf_path: Path) -> list:
    """Return [(defocus1, defocus2, dfhand), ...] in _CTF.txt row order
    (angle-sorted, matching _TLT.txt); caller zips this with parse_tlt()'s
    angle list to build an angle-keyed lookup."""
    rows = []
    with open(ctf_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append((float(parts[1]), float(parts[2]), int(float(parts[7]))))
    return rows


def _nearest_lookup(angle: float, table: dict):
    key = min(table, key=lambda a: abs(a - angle))
    if abs(key - angle) > ANGLE_MATCH_TOLERANCE:
        raise ValueError(f"no entry within {ANGLE_MATCH_TOLERANCE} deg of angle {angle}")
    return table[key]


def _write_star(path: Path, data_block_name: str, header_lines: list, rows: list):
    lines = ["# Written by relion5_star.py", "", f"data_{data_block_name}", "", "loop_"]
    lines += [f"{h} #{i+1}" for i, h in enumerate(header_lines)]
    for row in rows:
        lines.append("\t".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")


def _tilt_series_rows(images: list, tilt_axis_angle: float,
                       defocus_by_angle: dict) -> list:
    pre_exposure_by_angle = compute_pre_exposure_by_angle(images)
    rows = []
    for image in images:
        angle = image["tilt_angle"]
        pre = _nearest_lookup(angle, pre_exposure_by_angle)
        d1, d2 = _nearest_lookup(angle, defocus_by_angle)
        rows.append((
            image.get("movie_name", ""),
            image.get("frame_count", ""),
            angle,
            tilt_axis_angle,
            round(pre, 4),
            image.get("target_defocus", ""),
            d1, d2,
        ))
    return rows


def write_tilt_series_star(base: str, aretomo_out_dir: Path, mdoc_path: Path,
                            out_path: Path) -> dict:
    """engine: aretomo3 path. Write one tilt_series/<base>.star using
    AreTomo3's own per-image measured defocus (_CTF.txt) paired with the
    .mdoc's acquisition-order metadata. Returns {'dfhand', 'tilt_axis_angle'}
    for the caller's top-level tomograms.star row."""
    tlt_path = aretomo_out_dir / f"{base}_TLT.txt"
    ctf_path = aretomo_out_dir / f"{base}_CTF.txt"
    angles = prep.parse_tlt(tlt_path)
    ctf_rows = read_ctf_full(ctf_path)
    if len(angles) != len(ctf_rows):
        raise ValueError(
            f"{base}: _TLT.txt has {len(angles)} rows but _CTF.txt has {len(ctf_rows)} -- "
            f"can't pair them up")
    defocus_by_angle = {round(a, 1): (d1, d2) for a, (d1, d2, h) in zip(angles, ctf_rows)}
    dfhand = ctf_rows[0][2] if ctf_rows else None

    tilt_axis_angle, images = parse_mdoc(mdoc_path)
    rows = _tilt_series_rows(images, tilt_axis_angle, defocus_by_angle)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_star(out_path, base, TILT_SERIES_HEADER, rows)
    return {"dfhand": dfhand, "tilt_axis_angle": tilt_axis_angle}


def write_tilt_series_star_from_mdoc(base: str, mdoc_path: Path, out_path: Path) -> dict:
    """engine: aretomo2 (and imod_raptor, same mdoc-only mechanism) path. Write one tilt_series/<base>.star using
    ONLY the .mdoc's own per-image "Defocus" estimate for both
    rlnDefocusV/rlnDefocusU (no AreTomo3 CTF measurement exists for this
    path -- see module docstring). Returns {'tilt_axis_angle'} (no
    'dfhand' -- the caller must supply a configured constant)."""
    tilt_axis_angle, images = parse_mdoc(mdoc_path)
    defocus_by_angle = {
        round(image["tilt_angle"], 1): (image.get("defocus", ""), image.get("defocus", ""))
        for image in images
    }
    rows = _tilt_series_rows(images, tilt_axis_angle, defocus_by_angle)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_star(out_path, base, TILT_SERIES_HEADER, rows)
    return {"tilt_axis_angle": tilt_axis_angle}


def write_tomograms_star(cfg: dict, aretomo_out_dir: Path, mdoc_dir: Path,
                          out_dir: Path) -> Path:
    """engine: aretomo3. Write out_dir/tomograms.star plus out_dir/
    tilt_series/<base>.star for every tilt series found in
    aretomo_out_dir (same discovery logic as organize_for_isonet2.py's
    find_candidates). cfg needs: pix_size, at_bin (list, first value
    used), amp_contrast, kv, cs. Optional: optics_group_name (default
    'opticsGroup1')."""
    tilt_series_dir = out_dir / "tilt_series"
    optics_group_name = cfg.get("optics_group_name", "opticsGroup1")
    rows = []
    for base, evn, odd, full in prep.find_candidates(Path(aretomo_out_dir)):
        mdoc_path = Path(mdoc_dir) / f"{base}.mdoc"
        if not mdoc_path.exists():
            print(f"  [SKIP] {base}: no .mdoc file found at {mdoc_path}")
            continue
        star_name = f"{base}.star"
        meta = write_tilt_series_star(
            base, Path(aretomo_out_dir), mdoc_path, tilt_series_dir / star_name)
        rows.append((
            base, f"tilt_series/{star_name}",
            cfg["kv"], cfg["cs"], cfg["amp_contrast"],
            cfg["pix_size"], cfg["pix_size"], cfg["at_bin"][0],
            meta["dfhand"], optics_group_name,
        ))

    if not rows:
        raise RuntimeError(f"no tilt series with both AreTomo3 output and a .mdoc file "
                            f"found (checked {aretomo_out_dir} / {mdoc_dir})")

    out_path = out_dir / "tomograms.star"
    _write_star(out_path, "global", TOMOGRAMS_HEADER, rows)
    print(f"relion5_star: {len(rows)} tomograms -> {out_path}")
    return out_path


def write_tomograms_star_aretomo2(cfg: dict, tomogram_dir: Path, mdoc_dir: Path,
                                   out_dir: Path) -> Path:
    """engine: aretomo2 (and imod_raptor, same mdoc-only mechanism). Write out_dir/tomograms.star plus out_dir/
    tilt_series/<base>.star for every *<out_suffix>.mrc tomogram in
    tomogram_dir (excluding _proj preview files, same filter as
    run_pipeline.py's organize_aretomo2_output). cfg needs: pix_size,
    out_bin, amp_contrast, kv, cs, out_suffix. Optional:
    optics_group_name (default 'opticsGroup1'), defocus_handedness
    (default 1 -- no per-dataset measurement available for this path,
    see module docstring)."""
    tilt_series_dir = out_dir / "tilt_series"
    optics_group_name = cfg.get("optics_group_name", "opticsGroup1")
    dfhand = cfg.get("defocus_handedness", 1)
    suffix = cfg.get("out_suffix", "_AreTomo")
    rows = []
    for mrc in sorted(Path(tomogram_dir).glob(f"*{suffix}.mrc")):
        if "_proj" in mrc.stem:
            continue
        base = mrc.stem[: -len(suffix)] if mrc.stem.endswith(suffix) else mrc.stem
        mdoc_path = Path(mdoc_dir) / f"{base}.mdoc"
        if not mdoc_path.exists():
            print(f"  [SKIP] {base}: no .mdoc file found at {mdoc_path}")
            continue
        star_name = f"{base}.star"
        write_tilt_series_star_from_mdoc(base, mdoc_path, tilt_series_dir / star_name)
        rows.append((
            base, f"tilt_series/{star_name}",
            cfg["kv"], cfg["cs"], cfg["amp_contrast"],
            cfg["pix_size"], cfg["pix_size"], cfg["out_bin"],
            dfhand, optics_group_name,
        ))

    if not rows:
        raise RuntimeError(f"no tomograms with both an AreTomo2 output file and a .mdoc "
                            f"found (checked {tomogram_dir} / {mdoc_dir})")

    out_path = out_dir / "tomograms.star"
    _write_star(out_path, "global", TOMOGRAMS_HEADER, rows)
    print(f"relion5_star: {len(rows)} tomograms -> {out_path} (mdoc-only path)")
    return out_path
