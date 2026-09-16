#!/usr/bin/env python3
"""
Run AreTomo3 (motion correction + CTF + alignment + WBP reconstruction) on a
directory of raw tilt series, producing the *_EVN_Vol.mrc / *_ODD_Vol.mrc /
*_Vol.mrc / *_CTF.txt / *_TLT.txt files that organize_for_isonet2.py expects.

Flag names/defaults mirror aretomo.sh in this folder. Everything is driven
by a config dict (see pipeline_config.example.yaml, "aretomo:" section) --
edit the YAML, not this file, to change alignment/reconstruction behavior.

AreTomo3's flags use irregular casing (-PixSize, -kV, -McPatch, ...) so they
can't be auto-derived from snake_case config keys -- FLAG_MAP is the
translation table. Anything AreTomo3 supports that isn't in FLAG_MAP can
still be passed through the config's "extra" dict using the exact flag
spelling, e.g.:
    extra:
      "-OutXF": 1
"""
import subprocess
from pathlib import Path

FLAG_MAP = {
    "in_prefix": "-InPrefix",
    "in_suffix": "-InSuffix",
    "in_skips": "-InSkips",
    "out_dir": "-OutDir",
    "tmp_dir": "-TmpDir",
    "log_dir": "-LogDir",
    "gain": "-Gain",
    "pix_size": "-PixSize",
    "kv": "-kV",
    "cs": "-Cs",
    "fm_dose": "-FmDose",
    "eer_sampling": "-EerSampling",
    "fm_int": "-FmInt",
    "mc_patch": "-McPatch",
    "tilt_axis": "-TiltAxis",
    "tilt_cor": "-TiltCor",
    "at_patch": "-AtPatch",
    "align_z": "-AlignZ",
    "vol_z": "-VolZ",
    "at_bin": "-AtBin",
    "wbp": "-Wbp",
    "corr_ctf": "-CorrCTF",
    "out_imod": "-OutImod",
    "split_sum": "-SplitSum",
    "resume": "-Resume",
    "gpu": "-Gpu",
}

REQUIRED = ("in_prefix", "out_dir")


def build_command(cfg: dict, binary: str = "AreTomo3") -> list:
    missing = [k for k in REQUIRED if k not in cfg or cfg[k] is None]
    if missing:
        raise ValueError(f"aretomo config missing required key(s): {missing}")

    cmd = [binary]
    for key, flag in FLAG_MAP.items():
        if key not in cfg or cfg[key] is None:
            continue
        value = cfg[key]
        cmd.append(flag)
        if isinstance(value, (list, tuple)):
            cmd.extend(str(v) for v in value)
        else:
            cmd.append(str(value))

    for flag, value in (cfg.get("extra") or {}).items():
        cmd.append(flag)
        if isinstance(value, (list, tuple)):
            cmd.extend(str(v) for v in value)
        else:
            cmd.append(str(value))

    return cmd


def run_aretomo3(cfg: dict, binary: str = "AreTomo3", dry_run: bool = False) -> Path:
    for key in ("out_dir", "tmp_dir", "log_dir"):
        if cfg.get(key):
            Path(cfg[key]).mkdir(parents=True, exist_ok=True)

    cmd = build_command(cfg, binary=binary)
    print("+ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)

    return Path(cfg["out_dir"])
