# AreTomo3 → IsoNet2 pipeline

Takes raw cryo-ET tilt series through AreTomo3 (motion correction, CTF,
alignment, WBP reconstruction) and into IsoNet2 (missing-wedge correction /
denoising), driven by one YAML config. `run_pipeline.py` only decides what
order stages run in and wires each stage's output into the next stage's
input — every AreTomo3 and IsoNet2 parameter lives in the config, not in
code.

Pipeline stages: `aretomo` → `organize` → `prepare_star` → `deconv` →
`make_mask` → `refine` → `predict`.

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
| `aretomo_step.py` | Builds/runs the AreTomo3 CLI command |
| `isonet2_steps.py` | Thin wrappers around `isonet.py`'s subcommands |
| `organize_for_isonet2.py` | Organizes AreTomo3's EVN/ODD volumes into IsoNet2's expected layout |
| `pipeline_config.example.yaml` | Template config — copy to `pipeline_config.yaml` and edit |
| `config_editor.py` | Local web form for editing the config |
| `requirements.txt` | Python dependencies (just PyYAML) |

## Known gotchas

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
