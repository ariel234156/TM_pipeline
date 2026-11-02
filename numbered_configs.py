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
    ./pipeline edit N                       open config N in the web form (config_editor.py)
    ./pipeline run N [options]              run_pipeline.py on config N, e.g.
                                            ./pipeline run 4 --start-from refine

configs/ is writable by everyone with the sticky bit set (chmod 1777, like
/tmp): anyone can add a config, but only its owner can change or delete
it -- copy someone else's with `new --from N` instead. A run holds a lock
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


def cmd_new(args):
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", args.name).strip("_")
    if not name:
        sys.exit("give the config a name, e.g. ./pipeline new UriTomos_raw")
    src = resolve(args.source) if args.source else HERE / "pipeline_config.example.yaml"
    text = src.read_text()
    if text.startswith("# Pipeline run started"):
        # a run log (pipeline_params_*.txt): the config follows its header
        text = text.split("\n\n", 1)[1]
    CONFIG_DIR.mkdir(exist_ok=True)
    # one `new` at a time, so two people can't get the same number
    lock = _open_lock(Path("/tmp/cryoet-pipeline-new.lock"))
    fcntl.flock(lock, fcntl.LOCK_EX)
    number = max(numbered(), default=0) + 1
    path = CONFIG_DIR / f"{number:03d}_{name}.yaml"
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664), "w") as fh:
        fh.write(f"# Config {number:03d} ({name}): made by {getpass.getuser()} on "
                 f"{time.strftime('%Y-%m-%d %H:%M')} from {src.resolve()}\n" + text)
    os.close(lock)
    print(f"created {path}\n"
          f"edit it (./pipeline edit {number}, or any text editor), check it with "
          f"./pipeline run {number} --dry-run, then run it with ./pipeline run {number}")


def cmd_edit(args):
    config = resolve(args.config)
    if not os.access(config, os.W_OK):
        sys.exit(f"{config.name} belongs to {config.owner()} -- copy it to change it: "
                 f"./pipeline new <name> --from {args.config}")
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
