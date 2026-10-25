#!/usr/bin/env python3
"""
Overlay pytom's picked particles (from a RELION5-compat particles.star)
onto a central XY slice of each tomogram, one grid figure for the whole
batch -- a quick visual sanity check of template matching quality.
Runs as the pipeline's tm_figure stage, or standalone.

Coordinate convention (matches pytom_tm.extract's relion5_compat branch):
rlnCenteredCoordinateX/Y/ZAngst are centered on the tomogram's own
center and in Angstrom, so pixel = centered_angst / pixel_size + dim/2.
The pixel size is read per particle from rlnTomoTiltSeriesPixelSize --
despite the name, pytom fills it with the voxel size it actually matched
at (job.voxel_size), i.e. the tomogram's own spacing.

Usage:
    python3 visualize_picks.py --particles-star Template_matching/particles.star \\
        --tomogram-dir Corrected_tomo --out picks_overview.png
"""
import argparse
import re
from pathlib import Path


def find_tomogram(tomogram_dir: Path, tomo_name: str) -> Path:
    # IsoNet2 prefixes/suffixes the base name ("_isonet2_unet-medium_
    # <base>_final.mrc"). A plain substring match is wrong here:
    # "Position_1" is a substring of "Position_1_3" too (confirmed: the
    # old longest-match lookup drew Position_1's picks on Position_1_3's
    # slice). Require the name NOT to be followed by another "_<digit>"
    # or digit, and not preceded by an alphanumeric.
    pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(tomo_name) + r"(?!_?\d)")
    matches = [p for p in Path(tomogram_dir).glob("*.mrc") if pattern.search(p.stem)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one tomogram matching {tomo_name!r} in {tomogram_dir}, "
            f"found {[m.name for m in matches]}")
    return matches[0]


def make_overview(particles_star: Path, tomogram_dir: Path, out: Path,
                  z_slab_px: int = 20, ncols: int = 4) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mrcfile
    import numpy as np
    import starfile

    df = starfile.read(particles_star)
    tomo_names = sorted(df["rlnTomoName"].astype(str).unique())

    nrows = -(-len(tomo_names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.atleast_1d(axes).ravel()

    for ax, tomo_name in zip(axes, tomo_names):
        mrc_path = find_tomogram(tomogram_dir, tomo_name)
        with mrcfile.open(mrc_path, permissive=True) as m:
            vol = m.data
            nz, ny, nx = vol.shape
            cz = nz // 2
            img = np.array(vol[cz])

        sub = df[df["rlnTomoName"].astype(str) == tomo_name]
        pix = sub["rlnTomoTiltSeriesPixelSize"].to_numpy()
        px = sub["rlnCenteredCoordinateXAngst"].to_numpy() / pix + nx / 2
        py = sub["rlnCenteredCoordinateYAngst"].to_numpy() / pix + ny / 2
        pz = sub["rlnCenteredCoordinateZAngst"].to_numpy() / pix + nz / 2
        near = np.abs(pz - cz) <= z_slab_px

        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        ax.scatter(px[near], py[near], s=40, facecolors="none",
                   edgecolors="red", linewidths=1.0)
        ax.scatter(px[~near], py[~near], s=40, facecolors="none",
                   edgecolors="red", linewidths=0.6, alpha=0.25)
        ax.set_title(f"{tomo_name}  ({mrc_path.name})\n{len(sub)} particles "
                     f"({near.sum()} within {z_slab_px}px of z={cz}, solid)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(tomo_names):]:
        ax.axis("off")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles-star", required=True, type=Path)
    ap.add_argument("--tomogram-dir", required=True, type=Path)
    ap.add_argument("--z-slab-px", type=int, default=20,
                    help="particles within +/- this many pixels of the central Z slice are drawn solid, the rest faint")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    out = make_overview(args.particles_star, args.tomogram_dir, args.out, args.z_slab_px)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
