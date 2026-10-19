#!/usr/bin/env python3
"""
Generate a RELION5-compatible tomograms.star (+ one per-tomogram
tilt_series/<name>.star) from AreTomo3's own output (.mdoc, _CTF.txt,
_TLT.txt) -- matches the exact schema PyTom's own RELION5 reader expects
(pytom_tm.io.parse_relion5_star_data), verified 2026-10-19 by reading
that function's source directly rather than guessing from docs.

Schema (verified against pytom_tm's reader):
  tomograms.star (one row per tomogram):
    rlnTomoName, rlnTomoTiltSeriesStarFile, rlnTomoTiltSeriesPixelSize,
    rlnTomoTomogramBinning, rlnTomoHand, rlnAmplitudeContrast, rlnVoltage,
    rlnSphericalAberration
  tilt_series/<rlnTomoTiltSeriesStarFile> (one row per tilt image):
    rlnTomoNominalStageTiltAngle, rlnMicrographPreExposure, rlnDefocusV,
    rlnDefocusU

GOTCHA (confirmed empirically): AreTomo3's own _TLT.txt/_CTF.txt files are
sorted by TILT ANGLE, not acquisition order -- dose-symmetric collection
schemes are non-monotonic in angle (e.g. 0, 2, -2, -4, 4, 6, -6, ...), so
you cannot just accumulate dose row-by-row down those files. The .mdoc
file's own [ZValue = N] sections ARE in true acquisition order and each
carries its own ExposureDose, which is what's used here to compute
rlnMicrographPreExposure (cumulative dose strictly BEFORE each image),
matched back to the angle-sorted CTF/TLT rows by tilt angle.

This has NOT been validated by actually importing the output into
RELION5's own GUI/pipeliner -- only against PyTom's reader function,
which is a narrower bar (it only reads the 8 fields listed above; a real
RELION5 import job may expect additional columns this doesn't provide).
"""
from pathlib import Path

import organize_for_isonet2 as prep

ANGLE_MATCH_TOLERANCE = 0.1  # degrees


def parse_mdoc_acquisition_order(mdoc_path: Path) -> list:
    """Return [(tilt_angle, exposure_dose), ...] in true acquisition
    order, by reading .mdoc [ZValue = N] sections in file order (N=0,1,2...
    is the actual acquisition sequence, unlike AreTomo3's own _TLT.txt/
    _CTF.txt which are angle-sorted)."""
    entries = []
    tilt_angle = None
    with open(mdoc_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("[ZValue"):
                if tilt_angle is not None:
                    entries.append((tilt_angle, exposure_dose))
                tilt_angle = None
                exposure_dose = None
            elif line.startswith("TiltAngle"):
                tilt_angle = float(line.split("=")[1].strip())
            elif line.startswith("ExposureDose"):
                exposure_dose = float(line.split("=")[1].strip())
        if tilt_angle is not None:
            entries.append((tilt_angle, exposure_dose))
    return entries


def compute_pre_exposure_by_angle(acquisition_order: list) -> dict:
    """cumulative dose strictly BEFORE each image, keyed by its tilt
    angle (rounded for tolerant matching against CTF/TLT files)."""
    pre_exposure = {}
    cumulative = 0.0
    for tilt_angle, exposure_dose in acquisition_order:
        pre_exposure[round(tilt_angle, 1)] = cumulative
        cumulative += exposure_dose
    return pre_exposure


def _nearest_pre_exposure(angle: float, pre_exposure: dict) -> float:
    key = min(pre_exposure, key=lambda a: abs(a - angle))
    if abs(key - angle) > ANGLE_MATCH_TOLERANCE:
        raise ValueError(
            f"no acquisition-order angle within {ANGLE_MATCH_TOLERANCE} deg of {angle} "
            f"-- mdoc and _TLT.txt/_CTF.txt may be out of sync for this tilt series")
    return pre_exposure[key]


def read_ctf_full(ctf_path: Path) -> list:
    """Return [(defocus1, defocus2, dfhand), ...] for every row (angle-
    sorted, matching _TLT.txt row order -- see organize_for_isonet2.py's
    parse_ctf_defocus_near_zero for the same column-layout assumption)."""
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


def _write_star(path: Path, header_lines: list, rows: list):
    lines = ["# Written by relion5_star.py", "", "data_", "", "loop_"]
    lines += [f"{h} #{i+1}" for i, h in enumerate(header_lines)]
    for row in rows:
        lines.append("\t".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")


def write_tilt_series_star(base: str, aretomo_out_dir: Path, mdoc_path: Path,
                            out_path: Path) -> int:
    """Write one tilt_series/<base>.star. Returns the dfHand value (taken
    from the first row -- expected constant for one tilt series) so the
    caller can put it in the top-level tomograms.star row."""
    tlt_path = aretomo_out_dir / f"{base}_TLT.txt"
    ctf_path = aretomo_out_dir / f"{base}_CTF.txt"
    angles = prep.parse_tlt(tlt_path)
    ctf_rows = read_ctf_full(ctf_path)
    if len(angles) != len(ctf_rows):
        raise ValueError(
            f"{base}: _TLT.txt has {len(angles)} rows but _CTF.txt has {len(ctf_rows)} -- "
            f"can't pair them up")

    acquisition_order = parse_mdoc_acquisition_order(mdoc_path)
    pre_exposure = compute_pre_exposure_by_angle(acquisition_order)

    rows = []
    dfhand = None
    for angle, (d1, d2, h) in zip(angles, ctf_rows):
        pre = _nearest_pre_exposure(angle, pre_exposure)
        rows.append((angle, round(pre, 4), d1, d2))
        dfhand = h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_star(
        out_path,
        ["_rlnTomoNominalStageTiltAngle", "_rlnMicrographPreExposure",
         "_rlnDefocusV", "_rlnDefocusU"],
        rows,
    )
    return dfhand


def write_tomograms_star(cfg: dict, aretomo_out_dir: Path, mdoc_dir: Path,
                          out_dir: Path) -> Path:
    """Write out_dir/tomograms.star plus out_dir/tilt_series/<base>.star
    for every tilt series found in aretomo_out_dir (same discovery logic
    as organize_for_isonet2.py's find_candidates). cfg needs: pix_size,
    at_bin (list, first value used), amp_contrast, kv, cs."""
    tilt_series_dir = out_dir / "tilt_series"
    rows = []
    for base, evn, odd, full in prep.find_candidates(Path(aretomo_out_dir)):
        mdoc_path = Path(mdoc_dir) / f"{base}.mdoc"
        if not mdoc_path.exists():
            print(f"  [SKIP] {base}: no .mdoc file found at {mdoc_path}")
            continue
        star_name = f"{base}.star"
        dfhand = write_tilt_series_star(
            base, Path(aretomo_out_dir), mdoc_path, tilt_series_dir / star_name)
        rows.append((
            base, f"tilt_series/{star_name}",
            cfg["pix_size"], cfg["at_bin"][0], dfhand,
            cfg["amp_contrast"], cfg["kv"], cfg["cs"],
        ))

    if not rows:
        raise RuntimeError(f"no tilt series with both AreTomo3 output and a .mdoc file "
                            f"found (checked {aretomo_out_dir} / {mdoc_dir})")

    out_path = out_dir / "tomograms.star"
    _write_star(
        out_path,
        ["_rlnTomoName", "_rlnTomoTiltSeriesStarFile", "_rlnTomoTiltSeriesPixelSize",
         "_rlnTomoTomogramBinning", "_rlnTomoHand", "_rlnAmplitudeContrast",
         "_rlnVoltage", "_rlnSphericalAberration"],
        rows,
    )
    print(f"relion5_star: {len(rows)} tomograms -> {out_path}")
    return out_path
