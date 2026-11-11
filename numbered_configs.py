#!/usr/bin/env python3
"""
Numbered pipeline configs that anyone on this machine can list, copy, edit
and run, so several people can use the pipeline without all editing one
pipeline_config.yaml (on 2026-11-02 one user's config_editor session
switched another user's config to a different dataset mid-way).

Configs live in configs/ next to this script, as NNN_<name>.yaml. Use them
through the ./pipeline launcher, which picks a Python that has the
pipeline's packages (see pipeline.local):

    ./pipeline list                         number, owner, engine, data, who is running it
    ./pipeline new <name> [--from N|FILE]   copy the example (or config N, a config file
                                            or a run log) to the next free number
    ./pipeline edit N                       open config N in the web form (config_editor.py);
                                            its "Save as new numbered config" also makes a number
    ./pipeline run N [options]              run_pipeline.py on config N, e.g.
                                            ./pipeline run 4 --start-from refine

configs/ is writable by everyone with the sticky bit set (chmod 1777, like
/tmp): anyone can add a config, but only its owner can change or delete
it -- copy someone else's with `new --from N` (or open it with `edit N`,
which shows it read-only, and save it as a new number). A run holds a lock
on its config (a file in /tmp, released when the run ends however it ends),
so one config can't be run twice at the same time.
"""
import argparse
import fcntl
import getpass
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
NAME_RE = re.compile(r"^(\d{3})_(.+)\.yaml$")
# where each engine's raw data path is, for `list`
ENGINE_INPUT = {"aretomo3": ("aretomo", "in_prefix"), "aretomo2": ("aretomo2", "in_dir"),
                "imod_raptor": ("imod", "in_dir")}


def numbered() -> dict:
    """{number: path} of every NNN_<name>.yaml in configs/."""
    configs = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        m = NAME_RE.match(path.name)
        if m:
            configs[int(m.group(1))] = path
    return configs


def resolve(ref: str) -> Path:
    """A config number (4 or 004), or a path to a config file."""
    if ref.isdigit():
        configs = numbered()
        if int(ref) not in configs:
            sys.exit(f"no config number {int(ref)} in {CONFIG_DIR} -- see: ./pipeline list")
        return configs[int(ref)]
    if not Path(ref).is_file():
        sys.exit(f"{ref}: not a config number or a file")
    return Path(ref)


def _open_lock(path: Path, create: bool = True):
    """A lock file in /tmp that every user can lock: a run by one user has
    to lock a file another user created. None if create is False and the
    file doesn't exist."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT if create else os.O_RDONLY, 0o666)
    except FileNotFoundError:
        return None
    try:
        os.fchmod(fd, 0o666)  # the umask may have dropped group/other write
    except PermissionError:
        pass  # someone else's file: they already did this
    return fd


def _lock_path(config: Path) -> Path:
    digest = hashlib.sha1(str(config.resolve()).encode()).hexdigest()[:12]
    return Path("/tmp") / f"cryoet-pipeline-{digest}.lock"


def running(config: Path) -> str:
    """Who is running config right now ('' if nobody)."""
    fd = _open_lock(_lock_path(config), create=False)
    if fd is None:
        return ""
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return ""
    except BlockingIOError:
        return os.pread(fd, 200, 0).decode(errors="replace").strip() or "yes"
    finally:
        os.close(fd)


def cmd_list(args):
    configs = numbered()
    if not configs:
        print(f"no numbered configs in {CONFIG_DIR} yet -- make one: ./pipeline new <name>")
        return
    rows = [("no.", "name", "owner", "engine", "running", "raw data")]
    for number, path in configs.items():
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            cfg = {}
        engine = cfg.get("engine", "aretomo3")
        section, key = ENGINE_INPUT.get(engine, ("aretomo", "in_prefix"))
        data = cfg.get(section, {}).get(key) or cfg.get(section, {}).get("mrc_dir") or "?"
        rows.append((f"{number:03d}", NAME_RE.match(path.name).group(2), path.owner(),
                     engine, running(path) or "-", str(data)))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) + "  " + row[-1])


def create(name: str, text: str, src) -> tuple:
    """Write text as configs/NNN_<name>.yaml at the next free number, owned
    by whoever runs this (`new`, and config_editor.py's "Save as new
    numbered config"). Returns (number, path)."""
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_")
    if not name:
        raise ValueError("give the config a name, e.g. UriTomos_raw")
    CONFIG_DIR.mkdir(exist_ok=True)
    # one at a time, so two people can't get the same number
    lock = _open_lock(Path("/tmp/cryoet-pipeline-new.lock"))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        number = max(numbered(), default=0) + 1
        path = CONFIG_DIR / f"{number:03d}_{name}.yaml"
        with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664), "w") as fh:
            fh.write(f"# Config {number:03d} ({name}): made by {getpass.getuser()} on "
                     f"{time.strftime('%Y-%m-%d %H:%M')} from {src}\n" + text)
    finally:
        os.close(lock)
    return number, path


def cmd_new(args):
    src = resolve(args.source) if args.source else HERE / "pipeline_config.example.yaml"
    text = src.read_text()
    if text.startswith("# Pipeline run started"):
        # a run log (pipeline_params_*.txt): the config follows its header
        text = text.split("\n\n", 1)[1]
    try:
        number, path = create(args.name, text, src.resolve())
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"created {path}")
    set_machine_tools(path)
    print(f"edit it (./pipeline edit {number}, or any text editor), check it with "
          f"./pipeline run {number} --dry-run, then run it with ./pipeline run {number}")


# Where this machine's software is: set by install.sh in pipeline.local
# (exported), as config key -> environment variable. Software paths belong
# to the machine, not the dataset, so a config copied from another machine
# (or from the example) gets this machine's.
MACHINE_TOOLS = {"aretomo_bin": "PIPELINE_ARETOMO3_BIN", "aretomo2_bin": "PIPELINE_ARETOMO2_BIN",
                 "imod.bin_dir": "PIPELINE_IMOD_BIN_DIR", "pytom.bin_dir": "PIPELINE_PYTOM_BIN_DIR"}


def set_machine_tools(path: Path):
    from config_editor import set_values
    updates = {key: os.environ[var] for key, var in MACHINE_TOOLS.items() if os.environ.get(var)}
    if os.environ.get("PIPELINE_ISONET_PYTHON") and os.environ.get("PIPELINE_ISONET_SCRIPT"):
        updates["isonet_bin"] = [os.environ["PIPELINE_ISONET_PYTHON"], os.environ["PIPELINE_ISONET_SCRIPT"]]
    if not updates:
        return
    cfg = yaml.safe_load(path.read_text()) or {}

    def current(key):
        node = cfg
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        return node

    updates = {k: v for k, v in updates.items() if current(k) != v}
    missing = set_values(path, updates)
    for key, value in updates.items():
        if key not in missing:
            print(f"  this machine's software (pipeline.local): {key}: {value}")


def cmd_edit(args):
    config = resolve(args.config)
    if not os.access(config, os.W_OK):
        print(f"{config.name} belongs to {config.owner()} -- it opens read-only; put in your "
              f"own data and click 'Save as new numbered config' to get a number of your own",
              flush=True)  # execv below drops anything still buffered
    os.execv(sys.executable, [sys.executable, str(HERE / "config_editor.py"),
                              "--config", str(config)] + args.options)


def cmd_run(args):
    config = resolve(args.config)
    cmd = [sys.executable, str(HERE / "run_pipeline.py"), "--config", str(config)] + args.options
    if "--dry-run" in args.options:
        os.execv(sys.executable, cmd)  # writes nothing worth protecting
    lock = _open_lock(_lock_path(config))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"{config.name} is already running ({os.pread(lock, 200, 0).decode().strip()}) "
                 f"-- wait for it to finish, or copy it (./pipeline new <name> --from "
                 f"{args.config}) and give the copy its own output folders")
    os.ftruncate(lock, 0)
    os.pwrite(lock, f"{getpass.getuser()}, pid {os.getpid()}, since "
                    f"{time.strftime('%Y-%m-%d %H:%M')}\n".encode(), 0)
    # run_pipeline.py gets the lock's file descriptor too, so the config
    # stays locked while it runs even if this wrapper is killed
    sys.exit(subprocess.call(cmd, pass_fds=(lock,)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the numbered configs").set_defaults(func=cmd_list)
    p = sub.add_parser("new", help="copy a config to the next free number")
    p.add_argument("name", help="short name, e.g. the dataset: UriTomos_raw")
    p.add_argument("--from", dest="source",
                   help="config number, config file or run log (pipeline_params_*.txt) to copy "
                        "(default: pipeline_config.example.yaml)")
    p.set_defaults(func=cmd_new)
    p = sub.add_parser("edit", help="open a config in the web form")
    p.add_argument("config", help="config number")
    p.add_argument("options", nargs=argparse.REMAINDER, help="config_editor.py options (--port)")
    p.set_defaults(func=cmd_edit)
    p = sub.add_parser("run", help="run the pipeline on a config")
    p.add_argument("config", help="config number")
    p.add_argument("options", nargs=argparse.REMAINDER,
                   help="run_pipeline.py options (--start-from, --stop-after, --skip-denoise, --dry-run)")
    p.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
