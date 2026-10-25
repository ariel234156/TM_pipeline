# AreTomo3 / AreTomo2 / IMOD+RAPTOR → IsoNet2 pipeline

Takes cryo-ET tilt series through alignment + reconstruction, into IsoNet2
(missing-wedge correction / denoising), and optionally into PyTom (template
matching), driven by one YAML config. `run_pipeline.py` only decides what
order stages run in and wires each stage's output into the next stage's
input — every parameter lives in the config, not in code.

Pipeline stages: `aretomo` → `imod_align` → `imod_reconstruct` →
`relion5_star` → `organize` → `prepare_star` → `deconv` → `make_mask` →
`refine` → `predict` → `tm_template` → `tm_mask` → `tm_match` →
`tm_extract` → `tm_merge` → `tm_figure` (`imod_align`/`imod_reconstruct`,
`relion5_star` and the six `tm_*` stages are optional — only relevant for their
respective engine/feature, see below).

Three alignment engines are supported via the `engine` config key:
- **`aretomo3`** (default) runs AreTomo3 from raw frames + `.mdoc` files,
  producing EVN/ODD half-tomograms for IsoNet2's `isonet2-n2n`
  (noise2noise) method.
- **`aretomo2`** runs AreTomo2 from a directory of already-assembled tilt
  series `.mrc` stacks (e.g. motion-corrected by a separate tool) — no
  even/odd split, so it feeds IsoNet2's single-tomogram `isonet2` method
  instead (with `deconv` enabled).
- **`imod_raptor`** runs IMOD + RAPTOR fiducial-based alignment — a THIRD,
  independent engine, **not** chained through AreTomo at all. Needs real
  gold fiducial markers in the sample. Supports either raw EER frames +
  `.mdoc` (motion-corrected by IMOD's own `alignframes`, not AreTomo) or a
  directory of pre-assembled `.st`/`.mrc` stacks, via `imod.input_mode`.
  Like `aretomo2`, no even/odd split → IsoNet2's single-tomogram method.

## Prerequisites

- **AreTomo3** / **AreTomo2** / **IMOD (with RAPTOR)**, whichever engine(s)
  you use, and **IsoNet2**, installed separately on the machine you run
  this on — none of these are pip packages. `isonet.py` (IsoNet2's CLI)
  needs to be reachable, whether that's a bare command on `PATH` or a full
  path set in the config's `isonet_bin` key. Same for `aretomo_bin` /
  `aretomo2_bin` / `imod.bin_dir`.
- Python 3 with PyYAML: `pip install -r requirements.txt`.

## Quick start

```bash
git clone /mnt/cryonas02/Users/Gabriel_Frank/Ariel_Cohen/aretomo3-isonet2-pipeline.git
cd aretomo3-isonet2-pipeline
pip install -r requirements.txt
cp pipeline_config.example.yaml pipeline_config.yaml
```

Edit `pipeline_config.yaml` for your dataset (paths, pixel size, GPU IDs,
etc.) — either by hand, or with the included web form:

```bash
python3 config_editor.py
# open http://localhost:8765
```

`config_editor.py` is a small local web server (stdlib + PyYAML only, no
extra dependencies) covering every active parameter and raw path in the
config. It patches only the matched value text of each field's own line, so
it never disturbs the file's comments or structure. It edits the config
only — it does not run the pipeline. Binds to `localhost` only.

The `engine` field is a dropdown (AreTomo3 | AreTomo2 | IMOD+RAPTOR) —
picking one shows only that engine's section and hides the other two
(their values stay in the file untouched, just not shown).

Then, always check what's about to run before it actually runs:

```bash
python3 run_pipeline.py --config pipeline_config.yaml --dry-run
```

And for a real run:

```bash
python3 run_pipeline.py --config pipeline_config.yaml
```

Useful flags:

- `--start-from <stage>` / `--stop-after <stage>` — run only part of the
  pipeline (stage names above). Handy for resuming after a crash, or for
  running the IsoNet2 side against AreTomo3 output you already have:
  `--start-from organize`.
- `--dry-run` — print every command without executing it.

`refine` is a real GPU training job; run it under `tmux`/`screen` or in the
background rather than in a session that might get interrupted.

## Files

| File | What it does |
|---|---|
| `run_pipeline.py` | Orchestrator — chains all stages from one config |
| `aretomo_step.py` | Builds/runs the AreTomo3 CLI command (`engine: aretomo3`) |
| `aretomo2_step.py` | Batch-runs AreTomo2 over pre-assembled tilt series MRC stacks (`engine: aretomo2`) |
| `imod_step.py` | Fiducial-based alignment/reconstruction via IMOD + RAPTOR (`engine: imod_raptor`) |
| `isonet2_steps.py` | Thin wrappers around `isonet.py`'s subcommands |
| `organize_for_isonet2.py` | Organizes AreTomo3's EVN/ODD volumes into IsoNet2's expected layout |
| `relion5_star.py` | Generates a RELION5-compatible `tomograms.star` from any engine's output (optional, `relion5_star` stage) |
| `pytom_step.py` | Template matching via PyTom's pytom-match-pick toolkit (optional, `tm_*` stages) |
| `visualize_picks.py` | Picks-overview PNG (the `tm_figure` stage; also runnable on its own) |
| `pipeline_config.example.yaml` | Template config — copy to `pipeline_config.yaml` and edit |
| `config_editor.py` | Local web form for editing the config |
| `requirements.txt` | Python dependencies (just PyYAML) |

## IMOD + RAPTOR fiducial-based alignment (`engine: imod_raptor`)

A third, independent alignment engine for samples with real gold fiducial
markers — RAPTOR can't align without them. Two input modes
(`imod.input_mode`):

- `eer`: raw EER frames + a companion `.mdoc` per tilt series. Motion
  correction and per-tilt frame summing is done by IMOD's own
  `alignframes` tool (confirmed EER-capable), **not** AreTomo.
- `mrc`: a directory of pre-assembled tilt series `.st`/`.mrc` stacks,
  tilt angle already in each section's header.

From there (`imod_align` stage): `extracttilts` → `ccderaser -find`
(x-ray/hot-pixel removal, optional) → `newstack` (bin + normalize) →
`RAPTOR` (fiducial tracking + alignment) → optional gold erasure
(`imod.erase_gold.enabled`): the beads RAPTOR tracked plus every bead
`imodfindbeads` finds in each tilt image are erased with `ccderaser
-circle` on the **raw** binned stack (RAPTOR's `.fid.txt` is in raw-stack
coordinates), and that erased stack is then aligned with the same `.xf`.
Every tilt series is then reconstructed both ways (`imod_reconstruct`
stage): `tilt` WBP, optionally also a `-FakeSIRTiterations` variant and an
`mtffilter`-filtered variant, with and without gold. `imod.feed_variant`
picks which variant becomes the `<base><imod.out_suffix>.mrc` that
`organize`/`relion5_star` look for — always the gold-erased one when
erasure is enabled (a tilt series without a gold-erased stack is skipped,
never fed downstream with its gold).

Validated end-to-end on a real gold-fiducial ferritin dataset (bead-like
spots per aligned tilt image 50–66 → 2–12, ferritin signal at picked
particles unchanged). Use IMOD ≥ 5.1.13: 5.1.0's `tiltalign` crashes
intermittently inside RAPTOR. RAPTOR can still occasionally fail on a
tilt series — `imod_align` skips it, keeps going, and resumes on re-run
(it also re-erases gold from existing alignments without re-running
RAPTOR if only the gold-erased stack is missing).

## Template matching (optional)

The `tm_*` stages run PyTom's pytom-match-pick toolkit against the
IsoNet2-denoised tomograms from `predict`, given a reference density map
(MRC) and a particle diameter. See the `pytom:` section in
`pipeline_config.example.yaml` for the full set of options. Needs a
separate PyTom install (its own conda env, e.g. `pytom_tm`) — set
`pytom.bin_dir` to that env's `bin/` directory.

**Voxel size**: never trust the tomogram's own MRC header (IsoNet2's
`predict` has written a wrong one). `pytom.voxel_size_angstrom` must be the
tomogram's real spacing = tilt-series pixel size × tomogram binning (the
template and mask are built at it); `pytom.reference_map_voxel_size_angstrom`
is the reference map's own spacing. You'll see `WARNING: Provided voxel
size does not match voxel size annotated in tomogram/template mrc.` on
every run; that's expected.

**CTF/dose-weighted matching** (recommended with `relion5_star`): set
`pytom.relion5_tomograms_star` to the `tomograms.star` written by the
`relion5_star` stage. Matching then takes voxel size, tilt angles, dose and
per-tilt CTF straight from it (pytom's own `--relion5-tomograms-star`), and
refuses to run if that voxel size disagrees with `pytom.voxel_size_angstrom`.

**`tm_figure`** writes one PNG (`pytom.figure.out_name`, default
`picks_overview.png` in `pytom.out_dir`) with every tomogram's central XY
slice and its picks overlaid — check it before using the particles.

For RELION5-compatible output: set `"relion5-compat": true` under
`pytom.extract_candidates` to get `particles.star` with centered
coordinates in Angstrom (a real pytom flag). For a matching
`tomograms.star`, the optional `relion5_star` stage (before `organize`)
generates one — from AreTomo3's own `.mdoc`/`_CTF.txt`/`_TLT.txt` output
for `engine: aretomo3`, or from each tomogram's companion `.mdoc` alone
for `engine: aretomo2`/`imod_raptor` (neither does CTF estimation of its
own, so per-image defocus comes from the `.mdoc`'s own SerialEM-measured
value instead). The `engine: aretomo3` path's schema was verified against
both PyTom's own RELION5 reader and a real `relion_tomo_import SerialEM`
run (row order, pre-exposure convention, and column set all matched
exactly). The `imod_raptor` path writes everything RELION5's tomography
programs need (`rlnTomoTiltSeriesName` pointing at a full-resolution
aligned stack, `rlnTomoSizeX/Y/Z`, defocus in Å) and was verified with a
real `relion_tomo_subtomo` run (all picked particles extracted); the
`aretomo2` path has only been checked against PyTom's reader.

## Known gotchas

- **Never set `isonet_bin` to a bare command string if more than one
  `isonet.py` install exists on the machine.** This has broken real runs
  twice: a bare string depends on `PATH`, which depends on whatever
  shell/activation state happens to be active, and a stale install left
  reachable on `PATH` by an old dotfile entry can run far enough to look
  like it's working (imports fine, starts training) before failing on
  something unrelated. Pin the exact interpreter instead:
  `isonet_bin: ["/path/to/your/env/bin/python", "/path/to/isonet.py"]`.
- **`isonet.refine.method` must match the engine**: `isonet2-n2n` for
  `engine: aretomo3` (EVN/ODD halves), `isonet2` for `engine: aretomo2` or
  `imod_raptor` (full tomograms, no even/odd split). Using `isonet2-n2n`
  against a full-tomogram star file crashes trying to open `None` as a file (the
  half-tomogram columns aren't populated).
- **`isonet.refine.top_ctf_n`** (optional): caps training to only N
  tomograms instead of every one in the star file. For `engine: aretomo3`
  this picks the N tilt series with the highest CTF cross-correlation
  score (requires `organize` to have run at least once — it's what
  computes and persists the scores). For `engine: aretomo2`/`imod_raptor`
  neither does CTF estimation of its own, so it just takes the first N
  tomograms in star-file order instead — not a quality-ranked selection.
- **`deconv` is disabled by default.** With the default `engine: aretomo3`,
  this pipeline only ever organizes AreTomo3 output into EVN/ODD half-tomograms (no full tomogram), so
  `deconv`'s default input column stays unset and it would error. It also
  wouldn't help even if pointed at the halves: IsoNet2's `isonet2-n2n`
  training path reads the half-tomogram columns directly and ignores
  `deconv`'s output entirely. `deconv` only matters if you extend
  `organize` to also produce a full tomogram and use IsoNet2's
  single-tomogram `isonet2` method instead.
- **IsoNet2's own training loop is `for step in range(save_interval,
  epochs+1, save_interval)`.** If `epochs < save_interval` that range is
  empty — it silently trains for zero epochs, exits 0, and writes no
  checkpoint, with no error or warning. Keep `epochs >= save_interval`
  (ideally a multiple of it).
- **`refine`/`predict` can crash on exit with an "ICE default IO error
  handler" message** if matplotlib falls back to a Qt GUI backend on a
  machine with no real display — this happens *after* the actual
  checkpoints/volumes are already written, so it looks scarier than it is.
  `isonet2_steps.py` sets `MPLBACKEND=Agg` / `QT_QPA_PLATFORM=offscreen`
  to avoid it; if you still hit it, check whether something else in your
  environment is overriding those variables.
- **Verify the CTF column layout once on your own AreTomo3 output** before
  trusting `organize`'s automatic defocus extraction:
  `python3 organize_for_isonet2.py --in-dir <dir> --out-dir <dir>
  --show-ctf-sample <tilt_series_basename>`. It assumes the standard
  CTFFind4-style column layout (col2/col3 = defocus1/defocus2); this was
  confirmed correct against real AreTomo3 output but AreTomo3 doesn't
  document the format explicitly, so a future version could differ.
- **Check `AreTomo3 --help` / `isonet.py <subcommand> --help` on your own
  install before trusting an unfamiliar flag**, especially on AreTomo3:
  some versions can crash ungracefully (rather than erroring cleanly) on
  an unrecognized flag or malformed input, so `--dry-run` first and sanity
  check the printed command.
