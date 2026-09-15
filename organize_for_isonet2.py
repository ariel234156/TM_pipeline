#!/usr/bin/env python3
"""
Organize AreTomo3 output into the even/odd folder layout IsoNet2 expects,
and print the isonet.py prepare_star command to run next.

Run this on the same machine/filesystem as the AreTomo3 output (symlinks
are used by default, so it must be a local/POSIX filesystem, not SMB/CIFS).

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
    and that row order matches the companion *_TLT.txt row order.
    This has NOT been verified against real AreTomo3 output -- use
    --show-ctf-sample on one tilt series first and eyeball it.
    """
    if not ctf_path.exists():
        log.append((base, "WARN", f"no CTF file, defocus left as None: {ctf_path.name}"))
        return None
    if not tlt_path.exists():
        log.append((base, "WARN", f"no TLT file, defocus left as None: {tlt_path.name}"))
        return None

    angles = parse_tlt(tlt_path)
    rows = []
    with open(ctf_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[1]), float(parts[2])))
            except ValueError:
                continue

    if not rows or len(rows) != len(angles):
        log.append((base, "WARN",
                     f"CTF/TLT row count mismatch ({len(rows)} vs {len(angles)}), "
                     f"defocus left as None: {ctf_path.name}"))
        return None

    idx = min(range(len(angles)), key=lambda i: abs(angles[i]))
    d1, d2 = rows[idx]
    return round((d1 + d2) / 2.0, 1)


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

        evn_dst = evn_dir / f"{base}_EVN.mrc"
        odd_dst = odd_dir / f"{base}_ODD.mrc"
        link_or_copy(evn, evn_dst, args.copy)
        link_or_copy(odd, odd_dst, args.copy)

        included.append((base, evn_dst, odd_dst, defocus))
        log.append((base, "OK", "linked" if not args.copy else "copied"))

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
