# AreTomo3 → IsoNet2 pipeline

Takes cryo-ET tilt series through AreTomo (motion correction, CTF,
alignment, WBP reconstruction), into IsoNet2 (missing-wedge correction /
denoising), and optionally into PyTom (template matching), driven by one
YAML config. `run_pipeline.py` only decides what order stages run in and
wires each stage's output into the next stage's input — every parameter
lives in the config, not in code.

Pipeline stages: `aretomo` → `organize` → `prepare_star` → `deconv` →
`make_mask` → `refine` → `predict` → `tm_template` → `tm_mask` →
`tm_match` → `tm_extract` → `tm_merge` (the last five are optional,
template matching via PyTom — see below).

Two AreTomo engines are supported via the `aretomo_engine` config key:
**3** (default) runs AreTomo3 from raw frames + `.mdoc` files, producing
EVN/ODD half-tomograms for IsoNet2's `isonet2-n2n` (noise2noise) method.
**2** runs AreTomo2 from a directory of already-assembled tilt series
`.mrc` stacks (e.g. motion-corrected by a separate tool) — no even/odd
split, so it feeds IsoNet2's single-tomogram `isonet2` method instead
(with `deconv` enabled).

## Prerequisites

- **AreTomo3** and **IsoNet2** installed separately on the machine you run
  this on — they are not pip packages. `isonet.py` (IsoNet2's CLI) needs to
  be reachable, whether that's a bare command on `PATH` or a full path set
  in the config's `isonet_bin` key. Same for the `AreTomo3` binary and
  `aretomo_bin`.
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
| `aretomo_step.py` | Builds/runs the AreTomo3 CLI command (`aretomo_engine: 3`) |
| `aretomo2_step.py` | Batch-runs AreTomo2 over pre-assembled tilt series MRC stacks (`aretomo_engine: 2`) |
| `isonet2_steps.py` | Thin wrappers around `isonet.py`'s subcommands |
| `organize_for_isonet2.py` | Organizes AreTomo3's EVN/ODD volumes into IsoNet2's expected layout |
| `pytom_step.py` | Template matching via PyTom's pytom-match-pick toolkit (optional, `tm_*` stages) |
| `pipeline_config.example.yaml` | Template config — copy to `pipeline_config.yaml` and edit |
| `config_editor.py` | Local web form for editing the config |
| `requirements.txt` | Python dependencies (just PyYAML) |

## Template matching (optional)

The `tm_*` stages run PyTom's pytom-match-pick toolkit against the
IsoNet2-denoised tomograms from `predict`, given a reference density map
(MRC) and a particle diameter. See the `pytom:` section in
`pipeline_config.example.yaml` for the full set of options. Needs a
separate PyTom install (its own conda env, e.g. `pytom_tm`) — set
`pytom.bin_dir` to that env's `bin/` directory.

**Known gotcha**: IsoNet2's `predict` step does not write the tomogram's
real pixel size into its output MRC header (observed: header always says
1.0 Å/px). `pytom_step.py` always passes `voxel_size_angstrom` from the
config explicitly rather than trusting the tomogram's own header — make
sure that value matches your actual data. You'll see a `WARNING: Provided
voxel size does not match voxel size annotated in tomogram/template mrc.`
on every run because of this; that's expected.

## Known gotchas

- **Never set `isonet_bin` to a bare command string if more than one
  `isonet.py` install exists on the machine.** This has broken real runs
  twice: a bare string depends on `PATH`, which depends on whatever
  shell/activation state happens to be active, and a stale install left
  reachable on `PATH` by an old dotfile entry can run far enough to look
  like it's working (imports fine, starts training) before failing on
  something unrelated. Pin the exact interpreter instead:
  `isonet_bin: ["/path/to/your/env/bin/python", "/path/to/isonet.py"]`.
- **`isonet.refine.method` must match the AreTomo engine**: `isonet2-n2n`
  for `aretomo_engine: 3` (EVN/ODD halves), `isonet2` for `aretomo_engine:
  2` (full tomograms, no even/odd split). Using `isonet2-n2n` against a
  full-tomogram star file crashes trying to open `None` as a file (the
  half-tomogram columns aren't populated).
- **`isonet.refine.top_ctf_n`** (optional): train on only the N tilt
  series with the highest CTF cross-correlation score instead of every
  tomogram in the star file. Requires the `organize` stage to have run at
  least once (it's what computes and persists the scores).
- **`deconv` is disabled by default.** This pipeline only ever organizes
  AreTomo3 output into EVN/ODD half-tomograms (no full tomogram), so
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
