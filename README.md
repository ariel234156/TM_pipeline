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

## Installation

```bash
git clone https://github.com/ariel234156/TM_pipeline.git
cd TM_pipeline
./install.sh --check     # report what's there and what's missing; changes nothing
./install.sh             # asks where to install, then installs what's missing
```

On the lab machines, cloning the NAS copy is faster than GitHub:
`git clone /mnt/cryonas02/Users/Gabriel_Frank/Ariel_Cohen/aretomo3-isonet2-pipeline.git TM_pipeline`.

`install.sh` checks each piece of software the pipeline runs and installs
only what's missing, all into one folder, without root. It asks which
folder (Tab completes paths; Enter takes the default, `~/cryoet_software`),
unless you give one with `--prefix DIR`. With `--yes` it asks nothing and
uses `--prefix` or the default. Re-running is safe.

It first looks for copies that are already there, and uses one only if it
actually runs (and, for IMOD, is 5.1.13 or newer). It looks in:
- this machine's `pipeline.local` and the `--config` given
- `PATH` and the prefix
- every conda environment (`conda env list` and `~/.conda/environments.txt`)
- the usual software folders: `/opt`, `/usr/local`, your home, and
  `/home/*`, where lab machines often keep one folder per program
  (`/home/AreTomo2/AreTomo2`).

When several IsoNet2 installs share a version number, the one installed
from the tested commit wins.

| Component | Installed as | Version installed |
|---|---|---|
| Python for the pipeline (PyYAML, numpy, mrcfile, starfile, matplotlib, tifffile) | conda environment | Python 3.11 |
| IMOD, including RAPTOR (older than 5.1.13 counts as missing) | IMOD's own installer, from bio3d.colorado.edu | 5.1.13 |
| IsoNet2 | GitHub, plus its own `isonet2_environment.yml` | commit `6c5c7bb` |
| PyTom (`pytom-match-pick`) | conda environment (cupy) + pip | 0.9.0 |
| AreTomo3 | built from GitHub | 2.3.1 (`8a068cf`) |
| AreTomo2 | built from GitHub | v1.1.2 |

These are the versions the pipeline was tested with; flags such as
`--aretomo3-ref` or `--pytom-version` choose others (`./install.sh --help`).
Installing everything took about 20 minutes and 18 GB here, most of it for
IsoNet2's PyTorch/CUDA environment. Its checks and installs ignore Python
packages installed with `pip install --user` (`~/.local`), which can
otherwise break conda environments.

Afterwards it writes what it found or installed into `pipeline.local`, as
this machine's block (one per hostname, so several machines can share a
checkout on a network drive; rerunning replaces only this machine's
block):
- the pipeline's Python
- `PATH` additions for configs that call AreTomo by bare name
- the software paths, which `./pipeline new` puts into every config it
  makes

So after `./install.sh`, `./pipeline` works for every account on the
machine (see "Several users on one machine" below).

With `--config FILE` it also writes the paths into that config, keeping all
comments, and creates the file from `pipeline_config.example.yaml` if it
doesn't exist yet. The example has its `imod:` and `pytom:` sections
commented out; for those, it prints the lines to add instead. It also writes
two launchers that always use the right Python, into the install folder
(shown here for the default):

```bash
~/cryoet_software/bin/cryoet-pipeline --config my_config.yaml
~/cryoet_software/bin/cryoet-config-editor --config my_config.yaml
```

Useful options: `--only imod,pytom` or `--skip aretomo2` to limit the
components, `--cuda /usr/local/cuda-12.4` to choose the CUDA toolkit for
the AreTomo builds, and `--ignore-existing all` for a fully self-contained
install that ignores copies found elsewhere. Logs go to `logs/` in the
install folder.

On a slow or unreliable network, downloads resume where they broke off.
Git checkouts are partial clones: file contents come only for the commit
used, and IsoNet2's GUI app, tutorial images and preprint PDF are left out.
The conda environments (IsoNet2's PyTorch/CUDA one is ~10 GB) still need a
working connection.

**What it can't install** (ask an administrator): the NVIDIA driver; `git`,
`make`, and `curl` or `wget`; and, only for building AreTomo3/AreTomo2, a
CUDA toolkit and the system C++ compiler (`/usr/bin/g++`). If there is no
conda, it installs Miniforge into the prefix first. RELION isn't needed:
the pipeline writes RELION5 star files itself.

## Quick start

Edit `pipeline_config.yaml` for your dataset (paths, pixel size, GPU IDs,
etc.) — either by hand, or with the included web form:

```bash
~/cryoet_software/bin/cryoet-config-editor     # or: python3 config_editor.py
# prints e.g. "Open http://localhost:8766/?token=..." -- open that link
```

Several people can run the editor at once on the same machine: each one
takes the first free port in 8765–8770 (`--port 8767` or `--port 9000-9005`
to choose). Each person should edit their own numbered config
(`./pipeline edit N`, see below), not a shared one. Every account on the
machine can reach `localhost`, and the editor saves as whoever started it,
so the page only opens with the token from the printed link. A config you
can't write opens read-only, with the commands for a numbered copy of your
own. If two people have the same file open, a save from a page that's out
of date is refused instead of silently undoing the other person's changes;
reload and save again.

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

Every real run writes a copy of its config and command line to the
top-level `log_dir` (the run-log folder, for every engine) as
`pipeline_params_<date>_<time>.txt`. Configs without `log_dir` use
`aretomo.log_dir`, AreTomo3's own log folder, as before.

Useful flags:

- `--start-from <stage>` / `--stop-after <stage>` — run only part of the
  pipeline (stage names above). Handy for resuming after a crash, or for
  running the IsoNet2 side against AreTomo3 output you already have:
  `--start-from organize`.
- `--dry-run` — print every command without executing it. Writes nothing
  (no run log, folders or star files), but makes the checks that only read:
  the training tomograms against the current star file, and template
  matching's voxel size and tomogram names against
  `pytom.relion5_tomograms_star`.
- Template matching (`tm_*`) is skipped when the config has no `pytom:`
  section.
- `--skip-denoise` (or `skip_denoise: true` in the config) — skip IsoNet2
  (`prepare_star` to `predict`) and run template matching straight on
  `organize`'s tomograms: IMOD/AreTomo2 tomograms as they are, AreTomo3's
  full volumes rotated to Z along the beam into `<organize.out_dir>/FULL`.
  For AreTomo3, `organize` must run with it on (`--start-from organize
  --skip-denoise`). Use a separate `pytom.out_dir`, or the results
  overwrite those of a denoised run.

`refine` is a real GPU training job; run it under `tmux`/`screen` or in the
background rather than in a session that might get interrupted.

## Several users on one machine: numbered configs

Instead of everyone editing one `pipeline_config.yaml`, each dataset gets
its own numbered config in `configs/` (`001_HPT_test.yaml`,
`004_UriTomos.yaml`, ...), run through `./pipeline` from any account:

```bash
./pipeline list                          # number, owner, engine, raw data, who is running it
./pipeline new MyData --from 4           # copy config 4 (or a file, or a run log) to the next number
./pipeline edit 5                        # web form for config 5
./pipeline run 5 --dry-run               # check, then:
./pipeline run 5                         # any run_pipeline.py option works: --start-from refine ...
```

The web form's **Save as new numbered config** (name box at the bottom)
also makes a number: it saves the form's values as the next free number,
owned by you, and the page then edits that one. So the quickest way to
start on your own data is `./pipeline edit 4` (someone else's config opens
read-only), put in your paths, and save it as a new number.

`configs/` is writable by everyone with the sticky bit (`chmod 1777`, like
`/tmp`): anyone can add a config, only its owner can change or delete it,
so copy someone else's (`new --from N`, or Save as new in the form) and
give it your own output folders and `log_dir` (you usually can't write
into theirs). A
run locks its config (a file in `/tmp`), so one config can't run twice at
once.

`./pipeline` runs with the Python and `PATH` set in `pipeline.local`, next
to it (machine-specific, not in git, written by `./install.sh`; see
Installation):

```bash
PIPELINE_PYTHON=/path/to/env/bin/python3   # has the packages in requirements.txt
PATH="$PATH:/opt/AreTomo3"                 # tools the configs call by bare name
export PIPELINE_IMOD_BIN_DIR=/opt/imod_5.1.13/bin   # and the other software paths
```

`./pipeline new` copies those software paths (AreTomo3/AreTomo2 binaries,
IsoNet2's `[python, isonet.py]`, IMOD's and PyTom's `bin_dir`) into the
new config, also when it copies a config made on another machine, and
prints each one it changes. `isonet.py prepare_star` runs in the star
file's folder (`organize.out_dir`), since it also writes a `.to_node.json`
where it runs.

## Files

| File | What it does |
|---|---|
| `run_pipeline.py` | Orchestrator — chains all stages from one config |
| `aretomo_step.py` | Builds/runs the AreTomo3 CLI command (`engine: aretomo3`) |
| `aretomo2_step.py` | Batch-runs AreTomo2 over pre-assembled tilt series MRC stacks (`engine: aretomo2`) |
| `imod_step.py` | Fiducial-based alignment/reconstruction via IMOD + RAPTOR (`engine: imod_raptor`) |
| `isonet2_steps.py` | Thin wrappers around `isonet.py`'s subcommands |
| `organize_for_isonet2.py` | Organizes AreTomo3's EVN/ODD volumes into IsoNet2's expected layout (rotated to Z along the beam) and reads each tomogram's tilt range from its `.aln` |
| `relion5_star.py` | Generates a RELION5-compatible `tomograms.star` from any engine's output (optional, `relion5_star` stage) |
| `pytom_step.py` | Template matching via PyTom's pytom-match-pick toolkit (optional, `tm_*` stages) |
| `visualize_picks.py` | Picks-overview PNG (the `tm_figure` stage; also runnable on its own) |
| `find_gain_defects.py` | Writes an AreTomo3 `-DefectFile` from a gain reference (needs numpy + tifffile) |
| `pipeline_config.example.yaml` | Template config — copy to `pipeline_config.yaml` and edit |
| `config_editor.py` | Local web form for editing the config |
| `pipeline`, `numbered_configs.py` | List, copy, edit and run numbered configs from any account (see above) |
| `install.sh` | Checks for, and installs if missing, all the software above (see Installation) |
| `requirements.txt` | The pipeline's own Python dependencies (`install.sh` installs them too) |

## AreTomo3 tilt axis and camera defects (`engine: aretomo3`)

Two settings decide whether AreTomo3's tomograms are aligned at all.

**Tilt axis (`aretomo.tilt_axis`).** Don't copy the `.mdoc`'s
`TiltAxisAngle`. It describes SerialEM's image orientation, while AreTomo3
reads the raw EER frames, which on this microscope are about 90° rotated
from it (AreTomo3's developers call this "incompatible coordinate systems",
issue #11). With the `.mdoc` value locked (`[-92.67, -1]`) every tomogram
was misaligned: concentric rings in XZ and no coherent specimen layer.

To measure the axis for a new microscope or session, run one tilt series
with `tilt_axis: [0, 1]`. A first value of 0 makes AreTomo3 search the full
range; a second value ≥ 0 lets it refine and flip the axis 180° from its
CTF handedness check. Read the ROT column of its `.aln`, then lock that
value for the batch with `[<angle>, -1]`, so every tomogram gets the same
axis and handedness. Locking matters because AreTomo3 2.3.1's random seed
moves refined axes by up to ~1.5° between runs (fixed in 2.3.3, PR #73),
and refinement started from a wrong value wandered 38–114° in testing.
For the 2024-08-26 frames dataset the value is **179.2°**: a full search,
three different starting values, and IMOD patch tracking agree within 0.6°.

**Camera defects (`aretomo.defect_file`).** If the gain reference marks a
dead column with huge values, AreTomo3 multiplies them in, and every tilt
gets a bright line at a fixed detector position. AreTomo3's own hot-pixel
detector can miss it. Make a defect file from the gain with
`python3 find_gain_defects.py <gain> camera_defects.txt` and point
`defect_file` at it. On the 2024-08-26 gain this masks column 1048 and
improved the CTF fit from 11.0 to 9.4 Å.

**`align_z: 0`** lets AreTomo3 use the thickness it measures for each tilt
series, as the AreTomoLive paper recommends. Local alignment
(`at_patch: [4, 4]`) gave no measurable gain at bin 8 here and is 2.7×
slower, so it's left off.

How these were measured (Position_1): even/odd half-tomogram FSC over the
80–30 Å band, plus XY/XZ slices. Mdoc axis locked: 0.204; AreTomo3's own
axis: 0.265.

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
- **How IsoNet2 gets AreTomo3's volumes and tilt range.** AreTomo3 writes
  volumes with the slab thickness along Y (its default `-FlipVol 0`), but
  IsoNet2 assumes Z is the beam direction: its missing-wedge mask,
  `make_mask`'s `z_crop` and `predict`'s normalization all do. So
  `organize` writes the EVN/ODD halves rotated exactly as IMOD's
  `clip rotx` would (real files, not symlinks: ~0.5 GB per tomogram at
  bin 8), and the AreTomo3 output is never changed. `prepare_star` then
  sets each tomogram's `rlnTiltMin`/`rlnTiltMax` from its `.aln` (the
  angles AreTomo3 reconstructed with, including its measured tilt offset),
  not one shared `-60`/`60`. Checked on 2026-10-29 data: IsoNet2's mask
  matched the measured Fourier wedge with r ≈ 0.59 before, 0.82–0.93 after.
  IsoNet2's output, which template matching reads, is in the same Z-along-
  the-beam layout.
- **`isonet.refine.top_ctf_n`** (optional): caps training to only N
  tomograms instead of every one in the star file. For `engine: aretomo3`
  this picks the N tilt series with the highest CTF cross-correlation
  score (requires `organize` to have run at least once — it's what
  computes and persists the scores). For `engine: aretomo2`/`imod_raptor`
  neither does CTF estimation of its own, so it just takes the first N
  tomograms in star-file order instead — not a quality-ranked selection.
- **`isonet.refine.train_tomos`** (optional): names the tomograms to train
  on instead (e.g. `[Position_5, Position_13_2]`, file names without
  `.mrc` and `_EVN`/`_final`); wins over `top_ctf_n`. Use it to train on
  tomograms that show the structures you care about.
- **`isonet.make_mask.mode`**: `particles` (IsoNet2's default, density/std
  50/50, patch 4), `aggregates` (70/70, patch 8: each aggregate or fibre
  bundle becomes one smooth region, empty ice still left out; 41% of voxels
  kept instead of 26% on 2026-11-02 data) or `none` (no mask). The mask only
  decides where `refine` centres its training cubes — `predict` corrects
  the whole tomogram either way — so `make_mask` masks only the tomograms
  `refine` trains on. Re-run from `make_mask` after changing either option.
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
# TM_pipeline
