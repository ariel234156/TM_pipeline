#!/usr/bin/env python3
"""
Organize AreTomo3 output into the even/odd folder layout IsoNet2 expects,
and print the isonet.py prepare_star command to run next.

Run this on the same machine/filesystem as the AreTomo3 output (symlinks
are used by default, so it must be a local/POSIX filesystem, not SMB/CIFS).
Volumes in AreTomo3's default layout (thickness along Y) are written as
rotated copies instead -- IsoNet2 needs Z along the beam (see
volume_needs_rotx). Each tomogram's tilt range from its .aln is printed
in the log; prepare_star takes only one --tilt_min/--tilt_max, so set
rlnTiltMin/rlnTiltMax per row in the star file afterwards (run_pipeline.py
does this automatically).

Example (matching aretomo.sh in this conversation):

    python3 organize_for_isonet2.py \
        --in-dir  /mnt/cryonas02/Users/Gabriel_Frank/Ariel_Cohen/test_ferr/aretomo3_out \
        --out-dir /mnt/cryonas02/Users/Gabriel_Frank/Ariel_Cohen/test_ferr/isonet2_input \
        --raw-pixel-size 1.51 --bin-factor 4

Before trusting the extracted defocus values, verify the CTF file column
layout on one tilt series:

    python3 organize_for_isonet2.py --in-dir <dir> --out-dir <dir> \
        --show-ctf-sample <tilt_series_basename>
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import mrcfile
import numpy as np

EVN_SUFFIX = "_EVN_Vol.mrc"
ODD_SUFFIX = "_ODD_Vol.mrc"
FULL_SUFFIX = "_Vol.mrc"
SKIP_MARKERS = ("_2ND_Vol", "_3RD_Vol")


def find_candidates(in_dir: Path):
    """Yield (basename, evn_path, odd_path, full_path) for every *_EVN_Vol.mrc
    found, excluding the non-CTF-corrected _2ND_Vol/_3RD_Vol reconstructions."""
    for evn in sorted(in_dir.glob(f"*{EVN_SUFFIX}")):
        if any(marker in evn.name for marker in SKIP_MARKERS):
            continue
        base = evn.name[: -len(EVN_SUFFIX)]
        odd = in_dir / f"{base}{ODD_SUFFIX}"
        full = in_dir / f"{base}{FULL_SUFFIX}"
        yield base, evn, odd, full


def check_pair(base, evn, odd, full, log):
    if not odd.exists():
        log.append((base, "SKIP", f"missing odd volume: {odd.name}"))
        return False
    for p in (evn, odd):
        if p.stat().st_size == 0:
            log.append((base, "SKIP", f"zero-size file: {p.name}"))
            return False
    if not full.exists():
        log.append((base, "WARN", f"full volume not found (not required): {full.name}"))
    return True


def parse_tlt(tlt_path: Path):
    """Return tilt angles in file order (column 1 of *_TLT.txt)."""
    angles = []
    with open(tlt_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            angles.append(float(line.split()[0]))
    return angles


def parse_ctf_defocus_near_zero(ctf_path: Path, tlt_path: Path, log, base):
    """
    Parse an AreTomo3 *_CTF.txt file (documented only as "CTFFind4
    compatible") and return the defocus (Angstrom, averaged over defocus1/
    defocus2) for the image whose tilt angle is closest to 0 degrees.

    ASSUMES the standard CTFFind4 column layout:
        col1=index  col2=defocus1[A]  col3=defocus2[A]  col4=azimuth ...
        col5=phase shift[rad]  col6=cross correlation  col7=fit spacing[A]  col8=dfHand
    and that row order matches the companion *_TLT.txt row order. Verified
    2026-09-15 against real AreTomo3 output with --show-ctf-sample -- the
    column layout is correct.
    """
    row = _find_near_zero_ctf_row(ctf_path, tlt_path, log, base, "defocus")
    if row is None:
        return None
    d1, d2, _score = row
    return round((d1 + d2) / 2.0, 1)


def parse_ctf_score_near_zero(ctf_path: Path, tlt_path: Path, log, base):
    """
    Return the CTF cross-correlation score (column 6 of an AreTomo3
    *_CTF.txt file -- "cross correlation", the standard CTF fit-quality
    metric) for the image whose tilt angle is closest to 0 degrees.
    Higher means a cleaner CTF fit / better-quality data for that tilt
    series. Shares the same row-matching logic and column-layout
    assumption as parse_ctf_defocus_near_zero -- see that function's
    docstring for the caveat about it not being independently verified
    against AreTomo3's own documentation.
    """
    row = _find_near_zero_ctf_row(ctf_path, tlt_path, log, base, "CTF score")
    if row is None:
        return None
    _d1, _d2, score = row
    return score


def _find_near_zero_ctf_row(ctf_path: Path, tlt_path: Path, log, base, what: str):
    """Shared parsing for parse_ctf_defocus_near_zero/parse_ctf_score_near_zero:
    returns (defocus1, defocus2, cc_score) for the near-zero-tilt row, or
    None (logging a WARN) if the files are missing or row counts mismatch."""
    if not ctf_path.exists():
        log.append((base, "WARN", f"no CTF file, {what} left as None: {ctf_path.name}"))
        return None
    if not tlt_path.exists():
        log.append((base, "WARN", f"no TLT file, {what} left as None: {tlt_path.name}"))
        return None

    angles = parse_tlt(tlt_path)
    rows = []
    with open(ctf_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                rows.append((float(parts[1]), float(parts[2]), float(parts[5])))
            except ValueError:
                continue

    if not rows or len(rows) != len(angles):
        log.append((base, "WARN",
                     f"CTF/TLT row count mismatch ({len(rows)} vs {len(angles)}), "
                     f"{what} left as None: {ctf_path.name}"))
        return None

    idx = min(range(len(angles)), key=lambda i: abs(angles[i]))
    return rows[idx]


def show_ctf_sample(in_dir: Path, base: str):
    ctf_path = in_dir / f"{base}_CTF.txt"
    tlt_path = in_dir / f"{base}_TLT.txt"
    if not ctf_path.exists():
        print(f"No such file: {ctf_path}", file=sys.stderr)
        sys.exit(1)
    print(f"--- head of {ctf_path.name} ---")
    with open(ctf_path) as fh:
        for i, line in enumerate(fh):
            if i >= 15:
                break
            print(line.rstrip())
    print()
    log = []
    val = parse_ctf_defocus_near_zero(ctf_path, tlt_path, log, base)
    for entry in log:
        print("  ".join(map(str, entry)), file=sys.stderr)
    print(f"\nParsed near-zero-tilt defocus (avg of defocus1/defocus2): {val}")
    print("Confirm this matches the actual column layout before trusting --defocus-source ctf.")


def link_or_copy(src: Path, dst: Path, copy: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def parse_aln_tilt_range(aln_path: Path, log, base):
    """(min, max) of the tilt angles AreTomo3 actually reconstructed with:
    the TILT column (10th) of the .aln's per-view rows. Those angles
    already include the AlphaOffset AreTomo3 measured (-TiltCor 1; up to
    ~25 deg on this data) and leave out the views it dropped as dark, so
    they -- not the .mdoc's nominal -60/+60 -- are where each volume's
    missing wedge really is. IsoNet2's own prepare_star help asks for
    exactly this ("final tilt angle used for tomogram reconstruction from
    your .tlt or .aln file"). Returns None if the .aln is missing/empty."""
    if not aln_path.exists():
        log.append((base, "WARN", f"{aln_path.name} not found -- tilt range left at the config value"))
        return None
    tilts = []
    for line in aln_path.read_text().splitlines():
        if line.startswith("# Local Alignment"):
            break  # per-patch rows follow; the global table is complete
        fields = line.split()
        if line.startswith("#") or len(fields) < 10:
            continue
        tilts.append(float(fields[9]))
    if not tilts:
        log.append((base, "WARN", f"no view rows in {aln_path.name} -- tilt range left at the config value"))
        return None
    return min(tilts), max(tilts)


def volume_needs_rotx(path: Path) -> bool:
    """True if the volume is stored the way AreTomo3 writes it by default
    (-FlipVol 0): slab thickness along the file's Y axis, tilt axis along
    Z. IsoNet2 needs Z along the beam -- its missing-wedge mask
    (missing_wedge.mw3D), make_mask's z_crop and predict's normalization
    all assume it -- so such a volume has to be rotated before IsoNet2
    sees it."""
    with mrcfile.open(path, header_only=True, permissive=True) as m:
        nx, ny, nz = int(m.header.nx), int(m.header.ny), int(m.header.nz)
    return ny < nx and ny < nz


def write_rotx(src: Path, dst: Path):
    """Write src rotated about X into dst, voxel-for-voxel what IMOD's
    `clip rotx` (= `trimvol -rx`) writes (checked against IMOD 5.1.13), so
    handedness is kept and the .aln tilt angles keep their sign. dst is
    replaced via a temp file + os.replace and never written through: an
    earlier run may have left a symlink there that points at AreTomo3's
    own output."""
    with mrcfile.open(src, permissive=True) as m:
        data = np.ascontiguousarray(np.rot90(m.data, 1, axes=(0, 1)))
        vs = m.voxel_size
        voxel = (float(vs.x), float(vs.z), float(vs.y))
    tmp = dst.with_name(dst.name + ".part")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    with mrcfile.new(tmp) as out:
        out.set_data(data)
        out.voxel_size = voxel
    os.replace(tmp, dst)


def place_volume(src: Path, dst: Path, copy: bool) -> str:
    """Put src where IsoNet2 reads it: rotated to Z-along-the-beam if it is
    in AreTomo3's default layout (a real file, since the data change),
    otherwise symlinked/copied as before. A rotated copy newer than src is
    reused, so re-running organize doesn't rewrite every volume."""
    if not volume_needs_rotx(src):
        link_or_copy(src, dst, copy)
        return "copied" if copy else "linked"
    if (dst.is_file() and not dst.is_symlink()
            and dst.stat().st_mtime >= src.stat().st_mtime
            and not volume_needs_rotx(dst)):
        return "rotated (reused)"
    write_rotx(src, dst)
    return "rotated"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in-dir", required=True, type=Path, help="AreTomo3 -OutDir")
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="Destination root; EVN/ and ODD/ subfolders are created here")
    ap.add_argument("--pixel-size", type=float,
                     help="Effective tomogram pixel size in Angstrom (PixSize x AtBin). "
                          "If omitted, derive it from --raw-pixel-size and --bin-factor.")
    ap.add_argument("--raw-pixel-size", type=float,
                     help="AreTomo3 -PixSize value (movie pixel size, Angstrom)")
    ap.add_argument("--bin-factor", type=float,
                     help="First -AtBin value used for the primary (CTF-corrected) reconstruction")
    ap.add_argument("--defocus-source", choices=["ctf", "none"], default="ctf",
                     help="ctf (default): extract per-tilt-series defocus from AreTomo3's "
                          "_CTF.txt -- verify the format first with --show-ctf-sample. "
                          "none: leave defocus as None for isonet2 to handle manually.")
    ap.add_argument("--copy", action="store_true",
                     help="Copy tomogram files instead of symlinking (default: symlink)")
    ap.add_argument("--show-ctf-sample", metavar="BASENAME",
                     help="Print raw lines of one tilt series' _CTF.txt plus the parsed "
                          "defocus value, then exit -- use this to sanity-check the column "
                          "layout before trusting --defocus-source ctf.")
    args = ap.parse_args()

    if args.show_ctf_sample:
        show_ctf_sample(args.in_dir, args.show_ctf_sample)
        return

    if args.pixel_size is not None:
        pixel_size = args.pixel_size
    elif args.raw_pixel_size is not None and args.bin_factor is not None:
        pixel_size = args.raw_pixel_size * args.bin_factor
    else:
        ap.error("provide --pixel-size, or both --raw-pixel-size and --bin-factor")

    evn_dir = args.out_dir / "EVN"
    odd_dir = args.out_dir / "ODD"
    evn_dir.mkdir(parents=True, exist_ok=True)
    odd_dir.mkdir(parents=True, exist_ok=True)

    log = []
    included = []  # (base, evn_dst, odd_dst, defocus)

    for base, evn, odd, full in find_candidates(args.in_dir):
        if not check_pair(base, evn, odd, full, log):
            continue

        defocus = None
        if args.defocus_source == "ctf":
            ctf_path = args.in_dir / f"{base}_CTF.txt"
            tlt_path = args.in_dir / f"{base}_TLT.txt"
            defocus = parse_ctf_defocus_near_zero(ctf_path, tlt_path, log, base)

        tilt_range = parse_aln_tilt_range(args.in_dir / f"{base}.aln", log, base)

        evn_dst = evn_dir / f"{base}_EVN.mrc"
        odd_dst = odd_dir / f"{base}_ODD.mrc"
        how = place_volume(evn, evn_dst, args.copy)
        place_volume(odd, odd_dst, args.copy)

        included.append((base, evn_dst, odd_dst, defocus))
        log.append((base, "OK", how if tilt_range is None
                    else f"{how}; tilt range {tilt_range[0]:.2f} {tilt_range[1]:.2f}"))

    manifest_path = args.out_dir / "isonet2_prep_manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["basename", "status", "detail"])
        for row in log:
            writer.writerow(row)

    print(f"\n{len(included)} tilt series included, {sum(1 for r in log if r[1]=='SKIP')} skipped.")
    print(f"Manifest written to: {manifest_path}")
    if any(r[1] in ("SKIP", "WARN") for r in log):
        print("\nIssues found:")
        for base, status, detail in log:
            if status in ("SKIP", "WARN"):
                print(f"  [{status}] {base}: {detail}")

    if not included:
        print("\nNothing to hand to isonet.py -- nothing was included.", file=sys.stderr)
        sys.exit(1)

    defocus_list = [d if d is not None else 0 for _, _, _, d in included]
    any_missing_defocus = any(d is None for _, _, _, d in included)

    print("\nOrder used for --defocus (must match EVN/ODD directory listing order):")
    for base, _, _, d in included:
        print(f"  {base}: {d}")

    if any_missing_defocus:
        print("\nWARNING: some tilt series had no usable defocus and were filled with 0 "
              "as a placeholder -- fix these before running prepare_star, either by "
              "supplying real values or passing --defocus None to let isonet2 handle it.",
              file=sys.stderr)

    defocus_str = "[" + ",".join(str(d) for d in defocus_list) + "]"
    print("\nNext step -- run on the machine with isonet2 installed:\n")
    print(
        f"isonet.py prepare_star --even {evn_dir} --odd {odd_dir} "
        f"--pixel_size {pixel_size} --defocus \"{defocus_str}\""
    )


if __name__ == "__main__":
    main()
