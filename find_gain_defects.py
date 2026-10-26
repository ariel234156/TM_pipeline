#!/usr/bin/env python3
"""
Write an AreTomo3 -DefectFile (one "x y w h" rectangle per line) listing
the camera defects a gain reference marks with extreme values, so
AreTomo3 replaces those pixels with neighbouring good values in every
frame before motion correction.

Why this is needed (found on this dataset's EER gain reference,
20240826_170013_EER_GainReference.gain): column 1048 is a partly dead
detector column, and the gain marks it with values up to ~3800 (column
mean 67.9 vs ~1.0 everywhere else). AreTomo3 multiplies every frame by
the gain, so the few counts in that column come out 68-3800x too bright:
a bright dashed line at the same detector position in every tilt (66-101
robust SD above neighbouring columns). AreTomo3's own hot-pixel detector
misses it -- its 6-sigma threshold is computed from statistics the column
itself inflates ("Mean & Std: 2.85 20.65" in the log). A fixed pattern
like this pulls cross-correlation toward zero shift (IMOD's tiltxcorr man
page, "central peak exclusion": fixed noise from inadequate gain
normalization) and is backprojected as streaks.

Rule: a gain pixel is bad if it is <= 0 or more than --factor times
above/below the median. A column (row) with at least --min-line bad
pixels is written as one full-height (full-width) rectangle; other bad
pixels as 3x3 boxes. AreTomo3's frame rows run opposite to the TIFF row
order as tifffile reads it (measured on this dataset: the line's
brightness profile correlates 0.30 with the flipped gain column vs -0.07
unflipped), so isolated boxes are written at both y and ny-1-y -- a few
extra interpolated pixels, harmless either way. Full lines don't depend
on the orientation.

The rectangles are in unbinned camera pixels of the gain as given; if you
pass -RotGain/-FlipGain to AreTomo3, the defect coordinates must follow
the rotated/flipped orientation (not handled here).

Usage:
    python3 find_gain_defects.py 20240826_170013_EER_GainReference.gain camera_defects.txt
then set aretomo.defect_file to the output path.
"""
import argparse
from pathlib import Path


def load_gain(path: Path):
    import numpy as np
    if path.suffix.lower() in (".mrc", ".mrcs"):
        import mrcfile
        data = mrcfile.read(path)
        return np.asarray(data[0] if data.ndim == 3 else data, dtype=np.float64)
    import tifffile  # .gain (Thermo EER) and .tif/.tiff gain references are TIFF
    return np.asarray(tifffile.imread(path), dtype=np.float64)


def find_defects(gain, factor: float = 5.0, min_line: int = 16) -> list:
    import numpy as np
    ny, nx = gain.shape
    med = float(np.median(gain[gain > 0]))
    bad = (gain <= 0) | (gain > factor * med) | (gain < med / factor)
    entries = []
    cols = np.nonzero(bad.sum(0) >= min_line)[0]
    rows = np.nonzero(bad.sum(1) >= min_line)[0]
    entries += [(int(x), 0, 1, ny) for x in cols]
    entries += [(0, int(y), nx, 1) for y in rows]
    rest = bad.copy()
    rest[:, cols] = False
    rest[rows, :] = False
    for y, x in zip(*np.nonzero(rest)):
        for yy in sorted({int(y), ny - 1 - int(y)}):
            entries.append((max(int(x) - 1, 0), max(yy - 1, 0), 3, 3))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gain", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--factor", type=float, default=5.0,
                    help="bad if gain > factor*median or < median/factor (default 5)")
    ap.add_argument("--min-line", type=int, default=16,
                    help="bad pixels in one column/row to mask the whole line (default 16)")
    args = ap.parse_args()
    gain = load_gain(args.gain)
    entries = find_defects(gain, args.factor, args.min_line)
    args.out.write_text("".join(f"{x} {y} {w} {h}\n" for x, y, w, h in entries))
    full = sum(1 for e in entries if e[2] == gain.shape[1] or e[3] == gain.shape[0])
    print(f"{args.gain.name}: {len(entries)} defect rectangles ({full} full columns/rows) -> {args.out}")
    for e in entries[:20]:
        print("   ", *e)


if __name__ == "__main__":
    main()
