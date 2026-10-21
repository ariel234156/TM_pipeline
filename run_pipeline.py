#!/usr/bin/env python3
"""
End-to-end cryo-ET pipeline: raw tilt series -> AreTomo align+reconstruct ->
organize/prepare_star -> IsoNet2 deconv/make_mask/refine/predict denoising.

Two AreTomo engines are supported, set via the config's top-level
"aretomo_engine" key:
  3 (default): AreTomo3, from raw frames + .mdoc files, producing EVN/ODD
     half-tomograms -> organize_for_isonet2 even/odd layout -> IsoNet2's
     isonet2-n2n (noise2noise) method.
  2: AreTomo2 (aretomo2_step.py), from a directory of already-assembled
     tilt series .mrc stacks (one file per tilt series, e.g. produced by
     a separate motion-correction tool), one AreTomo2 call per file, no
     even/odd split -> prepare_star --full directly on AreTomo2's output
     dir -> IsoNet2's single-tomogram isonet2 method (with deconv enabled,
     since there's no even/odd split to feed isonet2-n2n otherwise).

Usage:
    python3 run_pipeline.py --config pipeline_config.yaml
    python3 run_pipeline.py --config pipeline_config.yaml --dry-run
    python3 run_pipeline.py --config pipeline_config.yaml --start-from refine
    python3 run_pipeline.py --config pipeline_config.yaml --stop-after deconv

Every tunable AreTomo3/AreTomo2 and IsoNet2 parameter lives in the YAML
config (copy pipeline_config.example.yaml and edit it) -- this script only
decides what order stages run in and wires each stage's output into the
next stage's input. It calls the AreTomo3/AreTomo2 binary and the isonet2
project's isonet.py CLI; all must be on PATH (or point at them with
config "isonet_bin" / "aretomo_bin" / "aretomo2_bin").

Requires PyYAML (`pip install pyyaml`).
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import yaml

import aretomo_step
import aretomo2_step
import isonet2_steps
import organize_for_isonet2 as prep
import pytom_step
import relion5_star

STAGES = ["aretomo", "relion5_star", "organize", "prepare_star", "deconv", "make_mask",
          "refine", "predict", "tm_template", "tm_mask", "tm_match", "tm_extract", "tm_merge"]


def log_run_params(cfg: dict, config_path: Path) -> Path:
    """Write a timestamped copy of the resolved config to a .txt file in
    aretomo.log_dir (falling back to the current directory), so every run
    leaves a record of exactly which parameters produced its output."""
    log_dir = cfg.get("aretomo", {}).get("log_dir")
    target_dir = Path(log_dir) if log_dir else Path(".")
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = target_dir / f"pipeline_params_{timestamp}.txt"
    header = (f"# Pipeline run started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"# Config source: {config_path.resolve()}\n\n")
    out_path.write_text(header + config_path.read_text())
    print(f"Parameters logged to: {out_path}")
    return out_path


def run_organize(cfg: dict, aretomo_out_dir: Path, dry_run: bool):
    """In-process equivalent of organize_for_isonet2.py's main(): reuses the
    same find_candidates/check_pair/parse_ctf_defocus_near_zero/link_or_copy
    functions so behavior stays identical, but returns the EVN/ODD dirs,
    pixel size and defocus list as Python objects for the next stage
    instead of just printing the follow-up command."""
    in_dir = Path(aretomo_out_dir)
    out_dir = Path(cfg["out_dir"])
    evn_dir = out_dir / "EVN"
    odd_dir = out_dir / "ODD"

    pixel_size = cfg.get("pixel_size")
    if pixel_size is None:
        raw = cfg.get("raw_pixel_size")
        binf = cfg.get("bin_factor")
        if raw is None or binf is None:
            raise ValueError(
                "organize config needs pixel_size, or both raw_pixel_size and bin_factor")
        pixel_size = raw * binf

    if dry_run:
        print(f"+ [dry-run] would organize {in_dir} -> {evn_dir}, {odd_dir}")
        return evn_dir, odd_dir, pixel_size, []

    evn_dir.mkdir(parents=True, exist_ok=True)
    odd_dir.mkdir(parents=True, exist_ok=True)

    log = []
    included = []  # (base, defocus, ctf_score), same order as evn_dir file creation
    defocus_source = cfg.get("defocus_source", "ctf")
    copy = cfg.get("copy", False)

    for base, evn, odd, full in prep.find_candidates(in_dir):
        if not prep.check_pair(base, evn, odd, full, log):
            continue
        defocus = None
        if defocus_source == "ctf":
            defocus = prep.parse_ctf_defocus_near_zero(
                in_dir / f"{base}_CTF.txt", in_dir / f"{base}_TLT.txt", log, base)
        # CTF score is collected independent of defocus_source -- it's what
        # isonet.refine.top_ctf_n (see main()) ranks tomograms by.
        ctf_score = prep.parse_ctf_score_near_zero(
            in_dir / f"{base}_CTF.txt", in_dir / f"{base}_TLT.txt", log, base)
        evn_dst = evn_dir / f"{base}_EVN.mrc"
        odd_dst = odd_dir / f"{base}_ODD.mrc"
        prep.link_or_copy(evn, evn_dst, copy)
        prep.link_or_copy(odd, odd_dst, copy)
        included.append((base, defocus, ctf_score))

    for base, status, detail in log:
        print(f"  [{status}] {base}: {detail}")
    if not included:
        raise RuntimeError(
            "organize step found nothing to hand to isonet2 -- check the AreTomo3 output dir")

    defocus_list = [d if d is not None else 0 for _, d, _ in included]
    if any(d is None for _, d, _ in included):
        print("WARNING: some tilt series had no usable defocus, filled with 0 -- "
              "see [WARN]/[SKIP] lines above", file=sys.stderr)

    # Written here (not just returned) so isonet.refine.top_ctf_n can read
    # it back even on a separate `--start-from refine` invocation that
    # doesn't re-run organize.
    scores_path = out_dir / "ctf_scores.csv"
    with open(scores_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "basename", "ctf_score"])
        for i, (base, _defocus, score) in enumerate(included, start=1):
            writer.writerow([i, base, "" if score is None else score])
    if any(s is None for _, _, s in included):
        print("WARNING: some tilt series had no usable CTF score (see [WARN]/[SKIP] lines "
              "above) -- they're excluded from isonet.refine.top_ctf_n ranking", file=sys.stderr)

    print(f"organize: {len(included)} tilt series -> {evn_dir}, {odd_dir}")
    print(f"organize: CTF scores written to {scores_path} (for isonet.refine.top_ctf_n)")
    return evn_dir, odd_dir, pixel_size, defocus_list


def organize_aretomo2_output(aretomo2_out_dir: Path, clean_dir: Path, suffix: str) -> Path:
    """AreTomo2's output directory mixes each real reconstructed tomogram
    (<stem><suffix>.mrc) with auxiliary .aln and <stem><suffix>_projXY/
    _projXZ.mrc preview files. isonet.py's prepare_star --full does NOT
    filter these out on its own -- confirmed empirically: it picks up the
    .aln file and the preview-projection volumes as if they were tomograms
    too, corrupting the resulting star file. So, same as organize_for_
    isonet2.py does for AreTomo3's EVN/ODD volumes, symlink just the real
    tomogram volumes into a clean directory instead of pointing --full at
    AreTomo2's raw output directly."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for vol in sorted(Path(aretomo2_out_dir).glob(f"*{suffix}.mrc")):
        if "_proj" in vol.stem:
            continue
        dst = clean_dir / vol.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(vol.resolve(), dst)
        count += 1
    if count == 0:
        raise RuntimeError(
            f"no *{suffix}.mrc tomograms found in {aretomo2_out_dir} to organize")
    print(f"organize: {count} AreTomo2 tomograms -> {clean_dir}")
    return clean_dir


def select_top_ctf_indices(organize_out_dir, n: int) -> set:
    """Read the ctf_scores.csv written by run_organize() and return the
    1-based rlnIndex values (matching the star file's row numbering) of
    the top-n tilt series by CTF cross-correlation score."""
    scores_path = Path(organize_out_dir) / "ctf_scores.csv"
    if not scores_path.exists():
        raise RuntimeError(
            f"isonet.refine.top_ctf_n is set but {scores_path} doesn't exist yet -- "
            f"run the organize stage at least once first (it's what computes CTF scores "
            f"per tilt series); re-run with --start-from organize")

    rows = []
    with open(scores_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["ctf_score"] == "":
                continue
            rows.append((int(row["index"]), float(row["ctf_score"])))
    if not rows:
        raise RuntimeError(f"no usable CTF scores found in {scores_path}")

    rows.sort(key=lambda r: r[1], reverse=True)
    return {idx for idx, _score in rows[:n]}


def select_first_n_indices(star_file: Path, n: int) -> set:
    """aretomo_engine: 2 fallback for isonet.refine.top_ctf_n: AreTomo2 does
    no CTF estimation of its own, so there is no ctf_scores.csv to rank by
    (unlike the aretomo_engine: 3 path -- see select_top_ctf_indices). Just
    takes the first n data rows in star-file order (the order
    organize_aretomo2_output() symlinked tomograms in, i.e. sorted
    filename order) instead of ranking by any quality metric."""
    indices = []
    for line in Path(star_file).read_text().splitlines():
        first_token = line.split(None, 1)[0] if line.split() else None
        try:
            idx = int(first_token)
        except (TypeError, ValueError):
            continue
        indices.append(idx)
    if not indices:
        raise RuntimeError(f"no data rows found in {star_file}")
    return set(indices[:n])


def write_filtered_star(star_file: Path, keep_indices: set, suffix: str) -> Path:
    """isonet.py's refine has no --tomo_idx flag (unlike deconv/make_mask/
    predict) -- the only way to restrict which tomograms it trains on is to
    give it a star file that only contains those rows. Keeps every header/
    loop_/column-definition line as-is; a line counts as a data row (kept
    only if its rlnIndex, the first whitespace-separated token, is in
    keep_indices) if that first token parses as an integer -- true for
    every real data row and nothing in the header, by construction of the
    STAR format isonet.py's starfile package writes."""
    out_lines = []
    for line in Path(star_file).read_text().splitlines():
        first_token = line.split(None, 1)[0] if line.split() else None
        try:
            idx = int(first_token)
        except (TypeError, ValueError):
            out_lines.append(line)  # header/comment/blank line -- keep as-is
            continue
        if idx in keep_indices:
            out_lines.append(line)

    out_path = Path(star_file).with_name(f"{Path(star_file).stem}{suffix}.star")
    out_path.write_text("\n".join(out_lines) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--start-from", choices=STAGES, default=STAGES[0])
    ap.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    ap.add_argument("--dry-run", action="store_true",
                     help="print every command that would run, without running it")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    log_run_params(cfg, args.config)
    start_idx = STAGES.index(args.start_from)
    stop_idx = STAGES.index(args.stop_after)
    if start_idx > stop_idx:
        ap.error("--start-from must come before or equal to --stop-after")

    def active(stage):
        return start_idx <= STAGES.index(stage) <= stop_idx

    aretomo_engine = cfg.get("aretomo_engine", 3)
    organize_cfg = cfg.get("organize", {})
    isonet_cfg = cfg.get("isonet", {})
    isonet_bin = cfg.get("isonet_bin", "isonet.py")
    aretomo_bin = cfg.get("aretomo_bin", "AreTomo3")
    aretomo2_bin = cfg.get("aretomo2_bin", "AreTomo2")

    # organize.out_dir doubles as where the star file lives, for both
    # engines -- for engine 2 it's where the clean (symlinked, aux-file-
    # free) tomogram directory goes (see organize_aretomo2_output()).
    aretomo_out_dir = Path(
        cfg["aretomo2"]["out_dir"] if aretomo_engine == 2 else cfg["aretomo"]["out_dir"])
    evn_dir = odd_dir = full_dir = None
    pixel_size = None
    defocus_list = None

    star_name = isonet_cfg.get("prepare_star", {}).get("star_name", "tomograms.star")
    star_file = Path(organize_cfg.get("out_dir", ".")) / star_name

    if active("aretomo"):
        if aretomo_engine == 2:
            aretomo_out_dir = aretomo2_step.run_aretomo2_batch(
                cfg["aretomo2"], binary=aretomo2_bin, dry_run=args.dry_run)
        else:
            aretomo_out_dir = aretomo_step.run_aretomo3(
                cfg["aretomo"], binary=aretomo_bin, dry_run=args.dry_run)

    if active("relion5_star"):
        # Optional: generate a RELION5-compatible tomograms.star (+ one
        # tilt_series/<name>.star per tomogram). Both engines need a
        # companion .mdoc per tomogram -- AreTomo2's input stacks must
        # have one sitting next to them for this to work (see
        # relion5_star.py's module docstring).
        r5_cfg = cfg.get("relion5_star", {})
        if aretomo_engine == 2:
            a2_cfg = cfg.get("aretomo2", {})
            r5_out_dir = Path(r5_cfg.get("out_dir", Path(a2_cfg["out_dir"]) / "relion5"))
            mdoc_dir = Path(r5_cfg.get("mdoc_dir", a2_cfg["in_dir"]))
            if args.dry_run:
                print(f"+ [dry-run] would write {r5_out_dir}/tomograms.star "
                      f"from {aretomo_out_dir} / {mdoc_dir} (aretomo_engine: 2)")
            else:
                relion5_star.write_tomograms_star_aretomo2(
                    {"pix_size": a2_cfg["pix_size"],
                     "out_bin": a2_cfg["out_bin"],
                     "amp_contrast": a2_cfg.get("amp_contrast", 0.07),
                     "kv": a2_cfg["kv"],
                     "cs": a2_cfg["cs"],
                     "out_suffix": a2_cfg.get("out_suffix", "_AreTomo"),
                     "optics_group_name": r5_cfg.get("optics_group_name", "opticsGroup1"),
                     "defocus_handedness": r5_cfg.get("defocus_handedness", 1)},
                    aretomo_out_dir, mdoc_dir, r5_out_dir)
        else:
            r5_out_dir = Path(r5_cfg.get("out_dir", Path(cfg["aretomo"]["out_dir"]) / "relion5"))
            mdoc_dir = Path(r5_cfg.get("mdoc_dir", cfg["aretomo"]["in_prefix"]))
            if args.dry_run:
                print(f"+ [dry-run] would write {r5_out_dir}/tomograms.star "
                      f"from {aretomo_out_dir} / {mdoc_dir}")
            else:
                relion5_star.write_tomograms_star(
                    {"pix_size": cfg["aretomo"]["pix_size"],
                     "at_bin": cfg["aretomo"]["at_bin"],
                     "amp_contrast": cfg["aretomo"].get("amp_contrast", 0.07),
                     "kv": cfg["aretomo"]["kv"],
                     "cs": cfg["aretomo"]["cs"],
                     "optics_group_name": r5_cfg.get("optics_group_name", "opticsGroup1")},
                    aretomo_out_dir, mdoc_dir, r5_out_dir)

    if active("organize"):
        if aretomo_engine == 2:
            # AreTomo2 produces one full tomogram per input file -- no
            # even/odd split, but its output dir also contains .aln and
            # preview-projection files that must be filtered out before
            # handing anything to prepare_star --full (see
            # organize_aretomo2_output's docstring).
            a2_cfg = cfg.get("aretomo2", {})
            clean_dir = Path(organize_cfg.get("out_dir", Path(aretomo_out_dir) / "tomograms"))
            if args.dry_run:
                print(f"+ [dry-run] would organize {aretomo_out_dir} -> {clean_dir}")
                full_dir = clean_dir
            else:
                full_dir = organize_aretomo2_output(
                    aretomo_out_dir, clean_dir, a2_cfg.get("out_suffix", "_AreTomo"))
            pixel_size = a2_cfg.get("pix_size", 1.0) * a2_cfg.get("out_bin", 1.0)
        else:
            evn_dir, odd_dir, pixel_size, defocus_list = run_organize(
                organize_cfg, aretomo_out_dir, args.dry_run)

    if active("prepare_star"):
        ps_cfg = dict(isonet_cfg.get("prepare_star", {}))
        if pixel_size is None or (aretomo_engine != 2 and (evn_dir is None or odd_dir is None)):
            raise RuntimeError(
                "prepare_star needs the organize stage's output -- run with "
                "--start-from organize (or earlier), it can't be skipped on its own")
        # isonet.py writes --star_name wherever it resolves (cwd if it's a
        # bare filename) -- pass an absolute path so it lands in the same
        # dir every later stage expects to find it in.
        star_file = Path(organize_cfg.get("out_dir", ".")) / ps_cfg.get("star_name", star_name)
        ps_cfg["star_name"] = str(star_file)
        if aretomo_engine == 2:
            # AreTomo2 writes no CTF.txt-equivalent file -- no automatic
            # per-tomogram defocus available, so defocus is omitted
            # entirely (isonet.py falls back to its own default) unless
            # the user set isonet.prepare_star.defocus explicitly.
            isonet2_steps.prepare_star(
                pixel_size, full_dir=full_dir,
                isonet_bin=isonet_bin, dry_run=args.dry_run, **ps_cfg)
        else:
            isonet2_steps.prepare_star(
                pixel_size, "[" + ",".join(str(d) for d in defocus_list) + "]",
                even_dir=evn_dir, odd_dir=odd_dir,
                isonet_bin=isonet_bin, dry_run=args.dry_run, **ps_cfg)

    if active("deconv"):
        dc_cfg = dict(isonet_cfg.get("deconv", {}))
        if dc_cfg.pop("enabled", True):
            isonet2_steps.deconv(star_file, isonet_bin=isonet_bin, dry_run=args.dry_run, **dc_cfg)

    if active("make_mask"):
        mm_cfg = dict(isonet_cfg.get("make_mask", {}))
        isonet2_steps.make_mask(star_file, isonet_bin=isonet_bin, dry_run=args.dry_run, **mm_cfg)

    refine_out_dir = None
    if active("refine"):
        rf_cfg = dict(isonet_cfg.get("refine", {}))
        top_ctf_n = rf_cfg.pop("top_ctf_n", None)
        refine_star_file = star_file
        if top_ctf_n:
            # refine has no --tomo_idx (unlike deconv/make_mask/predict) --
            # give it a star file containing only the selected rows instead.
            if aretomo_engine == 2:
                # AreTomo2 does no CTF estimation -- no ctf_scores.csv to
                # rank by, so just take the first N tomograms instead of
                # the top N by CTF score.
                keep = select_first_n_indices(star_file, int(top_ctf_n))
                print(f"refine: top_ctf_n={top_ctf_n} (aretomo_engine: 2 -- no CTF score "
                      f"available, using first {top_ctf_n} tomograms instead) -> "
                      f"training on rlnIndex {sorted(keep)}")
            else:
                keep = select_top_ctf_indices(organize_cfg["out_dir"], int(top_ctf_n))
                print(f"refine: top_ctf_n={top_ctf_n} -> training on rlnIndex {sorted(keep)} "
                      f"({star_file})")
            refine_star_file = write_filtered_star(star_file, keep, suffix="_top_ctf")
        refine_out_dir = isonet2_steps.refine(
            refine_star_file, isonet_bin=isonet_bin, dry_run=args.dry_run, **rf_cfg)

    if active("predict"):
        pr_cfg = dict(isonet_cfg.get("predict", {}))
        model = pr_cfg.pop("model", None)
        if model is None:
            search_dir = refine_out_dir or Path(
                isonet_cfg.get("refine", {}).get("output_dir", "isonet_maps"))
            candidates = (sorted(search_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
                          if search_dir.exists() else [])
            if not candidates:
                if args.dry_run:
                    model = search_dir / "<model selected after refine runs>"
                    print(f"predict: [dry-run] no checkpoint yet in {search_dir}, "
                          f"would pick the most recent *.pt at run time")
                else:
                    raise RuntimeError(
                        f"no isonet.predict.model set in config and no *.pt checkpoint found "
                        f"in {search_dir} -- set isonet.predict.model explicitly in the YAML")
            else:
                model = candidates[-1]
                print(f"predict: no model set in config, using most recent checkpoint: {model}")
        isonet2_steps.predict(star_file, model, isonet_bin=isonet_bin, dry_run=args.dry_run, **pr_cfg)

    pytom_cfg = cfg.get("pytom", {})
    tomogram_dir = Path(pytom_cfg.get(
        "tomogram_dir", isonet_cfg.get("predict", {}).get("output_dir", "corrected_tomos")))
    template_file = Path(pytom_cfg.get("out_dir", ".")) / "template.mrc"
    mask_file = Path(pytom_cfg.get("out_dir", ".")) / "mask.mrc"
    match_destinations = {}

    if active("tm_template"):
        template_file = pytom_step.create_template(pytom_cfg, args.dry_run)

    if active("tm_mask"):
        mask_file = pytom_step.create_mask(pytom_cfg, template_file, args.dry_run)

    if active("tm_match"):
        match_destinations = pytom_step.match_template_batch(
            pytom_cfg, template_file, mask_file, tomogram_dir, args.dry_run)

    if active("tm_extract"):
        if not match_destinations:
            # resuming from a separate invocation -- reconstruct
            # destinations from what's already on disk under out_dir/match
            match_root = Path(pytom_cfg.get("out_dir", ".")) / "match"
            match_destinations = {d.name: d for d in sorted(match_root.glob("*")) if d.is_dir()}
        pytom_step.extract_candidates_batch(pytom_cfg, match_destinations, args.dry_run)

    if active("tm_merge"):
        pytom_step.merge_stars(pytom_cfg, args.dry_run)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
