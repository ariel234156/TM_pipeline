#!/usr/bin/env python3
"""
Overlay pytom's picked particles (from a RELION5-compat particles.star)
onto a central XY slice of each tomogram, one grid figure for the whole
batch -- a quick visual sanity check of template matching quality before
committing to a given extract_candidates cut-off.

Coordinate convention (matches pytom_tm.extract's relion5_compat branch):
rlnCenteredCoordinateX/Y/ZAngst are centered on the tomogram's own
center and in Angstrom, so pixel = centered_angst / pixel_size + dim/2.

Usage:
    python3 visualize_picks.py --particles-star Template_matching/particles.star \\
        --tomogram-dir Corrected_tomo --pixel-size-angstrom 9.12 \\
        --out picks_overview.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import starfile


def find_tomogram(tomogram_dir: Path, tomo_name: str) -> Path:
    # IsoNet2 prefixes/suffixes the base name (e.g.
    # "_isonet2_unet-medium_<base>_final.mrc") -- match by substring
    # rather than assuming an exact filename, same convention used by
    # pytom_step.py's own rlnTomoName remap.
    matches = [p for p in Path(tomogram_dir).glob("*.mrc") if tomo_name in p.stem]
    if not matches:
        raise FileNotFoundError(f"no tomogram matching {tomo_name!r} in {tomogram_dir}")
    return max(matches, key=lambda p: len(p.stem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles-star", required=True, type=Path)
    ap.add_argument("--tomogram-dir", required=True, type=Path)
    ap.add_argument("--pixel-size-angstrom", required=True, type=float)
    ap.add_argument("--z-slab-px", type=int, default=20,
                     help="only show/mark particles within +/- this many pixels of the central Z slice")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = starfile.read(args.particles_star)
    tomo_names = sorted(df["rlnTomoName"].unique())

    ncols = 4
    nrows = -(-len(tomo_names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.atleast_1d(axes).ravel()

    for ax, tomo_name in zip(axes, tomo_names):
        mrc_path = find_tomogram(args.tomogram_dir, tomo_name)
        with mrcfile.open(mrc_path, permissive=True) as m:
            vol = m.data
        nz, ny, nx = vol.shape
        cz = nz // 2
        img = vol[cz]

        sub = df[df["rlnTomoName"] == tomo_name]
        px = sub["rlnCenteredCoordinateXAngst"].to_numpy() / args.pixel_size_angstrom + nx / 2
        py = sub["rlnCenteredCoordinateYAngst"].to_numpy() / args.pixel_size_angstrom + ny / 2
        pz = sub["rlnCenteredCoordinateZAngst"].to_numpy() / args.pixel_size_angstrom + nz / 2
        near = np.abs(pz - cz) <= args.z_slab_px

        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        ax.scatter(px[near], py[near], s=40, facecolors="none",
                   edgecolors="red", linewidths=1.0)
        ax.scatter(px[~near], py[~near], s=40, facecolors="none",
                   edgecolors="red", linewidths=0.6, alpha=0.25)
        ax.set_title(f"{tomo_name}\n{len(sub)} particles ({near.sum()} near z={cz})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(tomo_names):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
