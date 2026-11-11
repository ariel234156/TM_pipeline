#!/usr/bin/env bash
# Check for, and install if missing, everything this pipeline runs:
#   pipeline  a Python with PyYAML/numpy/mrcfile/starfile/matplotlib/tifffile
#   imod      IMOD (incl. RAPTOR)           - self-installer from bio3d.colorado.edu
#   isonet2   IsoNet2                       - GitHub + its own conda environment file
#   pytom     PyTom (pytom-match-pick)      - conda env (cupy) + pip
#   aretomo3  AreTomo3                      - built from GitHub with the CUDA toolkit
#   aretomo2  AreTomo2                      - built from GitHub with the CUDA toolkit
# A component counts as present if a working copy is found: in this
# machine's pipeline.local, the --config given, on PATH, in --prefix, in any
# conda environment, or in the usual software folders (/opt, /usr/local,
# /home/*). Only missing ones are installed, all into --prefix (no root
# needed). What it finds or installs is written into pipeline.local (for
# ./pipeline and the configs it makes) and into --config. Run with --help.
set -uo pipefail   # not -e: one component failing must not stop the others
# Packages from `pip install --user` (~/.local/lib/pythonX.Y) are added to
# every Python of that version, conda environments included. A stale numpy
# there broke a fresh environment in testing, and pip would count them as
# already installed. So checks, installs and the launchers below all ignore them.
export PYTHONNOUSERSITE=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$HOME/cryoet_software"
CONFIG=""
MODE=install
YES=0
ONLY=""
SKIP=""
IGNORE=""
CUDA_DIR=""
JOBS="$(nproc 2>/dev/null || echo 4)"

# Versions this pipeline was tested with (each can be overridden by a flag).
IMOD_VERSION=5.1.13            # 5.1.0's tiltalign truncates .xf files on real data
IMOD_URL=https://bio3d.colorado.edu/imod/AMD64-RHEL5
ISONET2_REPO=https://github.com/IsoNet-cryoET/IsoNet2.git
ISONET2_REF=6c5c7bb
PYTOM_VERSION=0.9.0
ARETOMO3_REPO=https://github.com/czimaginginstitute/AreTomo3.git
ARETOMO3_REF=8a068cf           # 2.3.1
ARETOMO2_REPO=https://github.com/czimaginginstitute/AreTomo2.git
ARETOMO2_REF=v1.1.2
MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
COMPONENTS=(pipeline imod isonet2 pytom aretomo3 aretomo2)

usage() {
    cat <<EOF
Usage: ./install.sh [options]

Checks each component (${COMPONENTS[*]}) and installs the missing ones
into --prefix. Re-running is safe: whatever is already there is kept.
The pipeline Python is always checked (it reads and writes the config).

Looks in: pipeline.local, --config, PATH, --prefix, every conda
environment, and /opt, /usr/local, \$HOME and /home/* (e.g. /home/AreTomo2/).
Then writes this machine's block in pipeline.local (Python, PATH and the
software paths), which ./pipeline uses and copies into the configs that
"./pipeline new" makes.

  --prefix DIR           where missing software goes (default: \$HOME/cryoet_software)
  --config FILE          also write the paths into this config (created from
                         pipeline_config.example.yaml if missing)
  --check                only report what is found or missing; change nothing
  --only LIST            only these components, comma-separated (e.g. imod,pytom)
  --skip LIST            leave these components out (e.g. aretomo2)
  --ignore-existing LIST don't use copies found outside --prefix for these
                         components ("all", or a list; "conda" counts too) --
                         gives a self-contained install in --prefix
  --cuda DIR             CUDA toolkit used to build AreTomo3/AreTomo2
                         (default: \$CUDA_HOME, nvcc on PATH, /usr/local/cuda)
  --jobs N               parallel build jobs (default: $JOBS)
  -y, --yes              don't ask before installing
  --imod-version V       (default $IMOD_VERSION)
  --isonet2-ref REF      git commit/tag (default $ISONET2_REF)
  --pytom-version V      (default $PYTOM_VERSION)
  --aretomo3-ref REF     git commit/tag (default $ARETOMO3_REF = 2.3.1)
  --aretomo2-ref REF     git commit/tag (default $ARETOMO2_REF)

Needs (not installed by this script -- they need an administrator):
an NVIDIA driver; git, make, curl or wget; and, only to build AreTomo3/
AreTomo2, a CUDA toolkit plus the system C++ compiler (/usr/bin/g++).
Conda is installed (Miniforge, into --prefix) if none is found.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift ;;
        --config) CONFIG="$2"; shift ;;
        --check) MODE=check ;;
        --only) ONLY="$2"; shift ;;
        --skip) SKIP="$2"; shift ;;
        --ignore-existing) IGNORE="$2"; shift ;;
        --cuda) CUDA_DIR="$2"; shift ;;
        --jobs) JOBS="$2"; shift ;;
        -y|--yes) YES=1 ;;
        --imod-version) IMOD_VERSION="$2"; shift ;;
        --isonet2-ref) ISONET2_REF="$2"; shift ;;
        --pytom-version) PYTOM_VERSION="$2"; shift ;;
        --aretomo3-ref) ARETOMO3_REF="$2"; shift ;;
        --aretomo2-ref) ARETOMO2_REF="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
PREFIX="$(realpath -m "$PREFIX")"   # resolved like the paths it is compared with
LOGDIR="$PREFIX/logs"
HOST="$(hostname -s)"

# This machine's current settings for ./pipeline, if any: its Python and
# software paths are tried first, and its PATH additions are searched.
LOCAL="$REPO/pipeline.local"
LOCAL_PY=""
if [ -f "$LOCAL" ]; then
    PIPELINE_PYTHON=""
    # shellcheck source=/dev/null
    . "$LOCAL" || printf 'WARNING: couldn'"'"'t read %s\n' "$LOCAL" >&2
    LOCAL_PY="$PIPELINE_PYTHON"
fi

# ---------------------------------------------------------------- helpers
declare -A STATUS WHERE NOTE
SET_ARETOMO_BIN="" SET_ARETOMO2_BIN="" SET_IMOD_BIN_DIR="" SET_PYTOM_BIN_DIR=""
SET_ISONET_PY="" SET_ISONET_SCRIPT=""
REASON=""

say()  { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
in_list() { [[ ",$2," == *",$1,"* ]]; }
wanted() { { [ -z "$ONLY" ] || in_list "$1" "$ONLY"; } && ! in_list "$1" "$SKIP"; }
ignored() { in_list all "$IGNORE" || in_list "$1" "$IGNORE"; }
inside_prefix() { [[ "$1" == "$PREFIX"/* ]]; }
# usable($component, $path): candidates outside --prefix are skipped when ignored
usable() { [ -n "$2" ] && { inside_prefix "$2" || ! ignored "$1"; }; }
version_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]; }
resolve() {  # command name or path -> absolute path of an executable
    local p="$1"
    [ -n "$p" ] || return 1
    [[ "$p" == */* ]] || p="$(command -v "$p" 2>/dev/null)" || return 1
    [ -x "$p" ] && [ ! -d "$p" ] && readlink -f "$p"
}
run_log() {  # run_log <component> <command...>: append output to the component's log
    local log="$LOGDIR/$1.log"; shift
    mkdir -p "$LOGDIR"
    printf '\n+ %s\n' "$*" >> "$log"
    "$@" >> "$log" 2>&1
}
fail_tail() { REASON="$1 (last lines of $LOGDIR/$2.log below)"; tail -n 15 "$LOGDIR/$2.log" | sed 's/^/    | /' >&2; }
size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }
download() {  # download <url> <file>
    # Resumes: on a slow link that keeps dropping (IMOD's 200 MB installer
    # took 145 min and many reconnects on one machine), each attempt carries
    # on from where the last one stopped instead of starting over.
    # Gives up after 10 attempts in a row that got nothing more.
    local want="" rc before stuck=0
    [ -s "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    say "   downloading $1"
    command -v curl >/dev/null &&
        want="$(curl -sIL "$1" | tr -d '\r' | awk 'tolower($1) == "content-length:" {n = $2} END {print n}')"
    while [ "$stuck" -lt 10 ]; do
        [ -n "$want" ] && [ "$(size_of "$2.part")" -ge "$want" ] && break
        before="$(size_of "$2.part")"
        if command -v curl >/dev/null; then
            curl -fsSL -C - --retry 5 --retry-delay 10 --speed-time 120 --speed-limit 1000 \
                 -o "$2.part" "$1"
        else
            wget -q -c -O "$2.part" "$1"
        fi
        rc=$?
        [ "$rc" = 0 ] && break
        [ "$rc" = 22 ] && [ "$(size_of "$2.part")" = 0 ] && return 1   # HTTP error (e.g. 404): no retry
        if [ "$(size_of "$2.part")" -gt "$before" ]; then stuck=0; else stuck=$((stuck + 1)); fi
        say "   ... $(( $(size_of "$2.part") >> 20 )) MB so far, reconnecting"
        sleep 10
    done
    [ -s "$2.part" ] && { [ -z "$want" ] || [ "$(size_of "$2.part")" -ge "$want" ]; } &&
        mv "$2.part" "$2"
}
git_checkout() {  # git_checkout <component> <repo url> <dir> <ref> [sparse-checkout patterns to leave out]
    # A partial clone: all commits, but file contents only for <ref> (and
    # not the paths left out), fetched again on failure -- full clones of
    # IsoNet2 kept breaking off ("early EOF") on a slow link.
    local comp="$1" url="$2" dir="$3" ref="$4" i
    shift 4
    if [ ! -d "$dir/.git" ]; then
        [ -e "$dir" ] && { REASON="$dir exists but isn't a git checkout -- move it away"; return 1; }
        for i in 1 2 3 4 5; do
            run_log "$comp" git clone --filter=blob:none --no-checkout "$url" "$dir" && break
            rm -rf "$dir"
            [ "$i" = 5 ] && { fail_tail "git clone failed" "$comp"; return 1; }
            sleep 10
        done
    fi
    if [ $# -gt 0 ]; then
        run_log "$comp" git -C "$dir" sparse-checkout set --no-cone '/*' "$@" ||
            { fail_tail "git sparse-checkout failed" "$comp"; return 1; }
    fi
    for i in 1 2 3 4 5; do
        run_log "$comp" git -C "$dir" checkout -q -f "$ref" && return 0
        sleep 10
    done
    fail_tail "git checkout $ref failed" "$comp"
    return 1
}

# ---------------------------------------------------------------- searching
# Places shared software usually lives. On lab machines that is often a
# folder per program in /home (/home/AreTomo2/AreTomo2, /home/relion, ...).
SEARCH_ROOTS=("$PREFIX" "$HOME" /opt /usr/local /home/*)
search() {  # search <glob relative to a root>...: existing matches under every root
    local root pat f
    for root in "${SEARCH_ROOTS[@]}"; do
        [ -d "$root" ] || continue
        for pat in "$@"; do
            for f in "$root"/$pat; do [ -e "$f" ] && printf '%s\n' "$f"; done
        done
    done | awk '!seen[$0]++'
}
ENV_BINS=()
ENV_BINS_DONE=0
env_bins() {  # fills ENV_BINS: bin/ of every conda environment found (once)
    [ "$ENV_BINS_DONE" = 1 ] && return 0
    ENV_BINS_DONE=1
    local d
    find_conda
    # environments.txt: every environment this account's conda made,
    # wherever it is (also with -p), even when conda itself isn't on PATH
    mapfile -t ENV_BINS < <(
        { [ -n "$CONDA" ] && "$CONDA" env list 2>/dev/null | awk '!/^#/ && NF {print $NF}'
          cat "$HOME/.conda/environments.txt" 2>/dev/null
          printf '%s\n' "$PREFIX"/envs/* "$PREFIX"/*/env "$HOME"/.conda/envs/* \
                        "$HOME"/{miniforge3,miniconda3,anaconda3}/envs/*; } |
        while read -r d; do [ -d "$d/bin" ] && readlink -f "$d/bin"; done | awk '!seen[$0]++')
}

# ---------------------------------------------------------------- conda
CONDA=""
find_conda() {
    local c
    if ! ignored conda; then
        # on PATH, or in a usual place: also a shared one such as /home/anaconda3
        for c in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null)" "$HOME/miniforge3/bin/conda" \
                 "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda \
                 $(search 'bin/conda' '*conda*/bin/conda' '*forge*/bin/conda'); do
            [ -n "$c" ] && [ -x "$c" ] && { CONDA="$c"; break; }
        done
    fi
    [ -z "$CONDA" ] && [ -x "$PREFIX/miniforge3/bin/conda" ] && CONDA="$PREFIX/miniforge3/bin/conda"
    [ -n "$CONDA" ] && [ -x "$(dirname "$CONDA")/mamba" ] && CONDA="$(dirname "$CONDA")/mamba"
    [ -n "$CONDA" ]
}
ensure_conda() {
    find_conda && return 0
    say "   no conda found -- installing Miniforge into $PREFIX/miniforge3"
    download "$MINIFORGE_URL" "$PREFIX/downloads/Miniforge3-Linux-x86_64.sh" &&
        run_log conda bash "$PREFIX/downloads/Miniforge3-Linux-x86_64.sh" -b -p "$PREFIX/miniforge3" ||
        { fail_tail "Miniforge install failed" conda; return 1; }
    find_conda
}

# ---------------------------------------------------------------- pipeline python
PY_DEPS="import yaml, numpy, mrcfile, starfile, matplotlib, tifffile"
PIPELINE_PY=""
pipeline_ok() { [ -x "$1" ] && "$1" -c "$PY_DEPS" >/dev/null 2>&1; }
check_pipeline() {
    local c
    env_bins
    for c in "$LOCAL_PY" "$PREFIX/envs/cryoet_pipeline/bin/python" "$(command -v python3 2>/dev/null)" \
             "${ENV_BINS[@]/%//python3}"; do
        usable pipeline "$c" && pipeline_ok "$c" || continue
        PIPELINE_PY="$c"; WHERE[pipeline]="$c"
        case " ${ENV_BINS[*]/%//python3} " in
            *" $c "*) [ "$c" = "$LOCAL_PY" ] ||
                NOTE[pipeline]="another program's conda environment (has the packages); --ignore-existing pipeline makes one of its own" ;;
        esac
        return 0
    done
    REASON="no python3 with: ${PY_DEPS#import }"
    return 1
}
install_pipeline() {
    ensure_conda || return 1
    run_log pipeline "$CONDA" create -y -p "$PREFIX/envs/cryoet_pipeline" --override-channels -c conda-forge \
        python=3.11 pyyaml numpy mrcfile starfile matplotlib tifffile ||
        { fail_tail "conda create failed" pipeline; return 1; }
}

# Values currently in the config (read with the pipeline's own Python).
CFG_ARETOMO3="" CFG_ARETOMO2="" CFG_IMOD_BIN="" CFG_PYTOM_BIN=""
CFG_ISONET_PY="" CFG_ISONET_SCRIPT="" CFG_ISONET_CMD=""
read_config() {
    [ -f "$CONFIG" ] && [ -n "$PIPELINE_PY" ] || return 0
    eval "$("$PIPELINE_PY" - "$CONFIG" <<'PY'
import shlex, sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
iso = c.get("isonet_bin")
pair = isinstance(iso, list) and len(iso) == 2
vals = {
    "CFG_ARETOMO3": c.get("aretomo_bin"), "CFG_ARETOMO2": c.get("aretomo2_bin"),
    "CFG_IMOD_BIN": (c.get("imod") or {}).get("bin_dir"),
    "CFG_PYTOM_BIN": (c.get("pytom") or {}).get("bin_dir"),
    "CFG_ISONET_PY": iso[0] if pair else None, "CFG_ISONET_SCRIPT": iso[1] if pair else None,
    "CFG_ISONET_CMD": iso if isinstance(iso, str) else None,
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v or ''))}")
PY
)"
}

# ---------------------------------------------------------------- IMOD
imod_ok() {  # imod_ok <bin dir>
    local b="$1" d p v
    d="$(dirname "$b")"
    for p in RAPTOR tiltalign newstack tilt ccderaser imodfindbeads xfproduct point2model; do
        [ -x "$b/$p" ] || { REASON="$b/$p missing"; return 1; }
    done
    v="$(cat "$d/VERSION" 2>/dev/null || echo 0)"
    version_ge "$v" "$IMOD_VERSION" ||
        { REASON="IMOD $v at $d is older than $IMOD_VERSION (5.1.0's tiltalign truncates .xf files)"; return 1; }
    IMOD_DIR="$d" PATH="$b:$PATH" timeout 60 "$b/RAPTOR" -h >/dev/null 2>&1 ||
        { REASON="$b/RAPTOR -h failed"; return 1; }
    NOTE[imod]="IMOD $v"
}
check_imod() {
    local b first="" found
    # newest version first among the ones found by searching
    mapfile -t found < <(search 'imod_*/bin' 'IMOD/bin' 'IMOD_*/bin' 'IMOD/IMOD/bin' |
                         awk -F/ '{print $(NF-1) "\t" $0}' | sort -t$'\t' -k1,1 -rV | cut -f2)
    for b in "$CFG_IMOD_BIN" "${PIPELINE_IMOD_BIN_DIR:-}" "${IMOD_DIR:+$IMOD_DIR/bin}" \
             "$(dirname "$(command -v RAPTOR 2>/dev/null || echo /none/x)")" "$PREFIX/imod_$IMOD_VERSION/bin" \
             "${found[@]}"; do
        usable imod "$b" || continue
        if imod_ok "$b"; then
            WHERE[imod]="$b"
            [ "$b" = "$CFG_IMOD_BIN" ] || SET_IMOD_BIN_DIR="$b"
            return 0
        fi
        first="${first:-$REASON}"
    done
    REASON="${first:-no IMOD found}"
    return 1
}
install_imod() {
    local f="imod_${IMOD_VERSION}_RHEL8-64_CUDA12.0.sh"
    download "$IMOD_URL/$f" "$PREFIX/downloads/$f" || { REASON="download of $IMOD_URL/$f failed"; return 1; }
    # -name replaces only a previous copy of this same version in --prefix;
    # -skip leaves /etc/profile.d alone (the pipeline sets IMOD_DIR itself).
    run_log imod bash "$PREFIX/downloads/$f" -dir "$PREFIX" -name "imod_$IMOD_VERSION" -skip -yes ||
        { fail_tail "IMOD installer failed" imod; return 1; }
}

# ---------------------------------------------------------------- IsoNet2
isonet_ok() {  # isonet_ok <python> <isonet.py>  (or <command> "")
    local out
    if [ -n "$2" ]; then
        [ -x "$1" ] && [ -f "$2" ] || { REASON="$1 or $2 missing"; return 1; }
        out="$(MPLBACKEND=Agg timeout 300 "$1" "$2" --help 2>&1)"
    else
        out="$(MPLBACKEND=Agg timeout 300 "$1" --help 2>&1)"
    fi || { REASON="isonet.py --help failed: $(tail -n1 <<<"$out")"; return 1; }
}
ISONET_PY="" ISONET_SCRIPT=""   # the [python, isonet.py] pair found, for pipeline.local
check_isonet2() {
    local cmd py sc b
    if [ -n "$CFG_ISONET_PY" ] && usable isonet2 "$CFG_ISONET_PY" && isonet_ok "$CFG_ISONET_PY" "$CFG_ISONET_SCRIPT"; then
        WHERE[isonet2]="$CFG_ISONET_SCRIPT ($CFG_ISONET_PY)"
        ISONET_PY="$CFG_ISONET_PY"; ISONET_SCRIPT="$CFG_ISONET_SCRIPT"; return 0
    fi
    if [ -z "$CFG_ISONET_PY" ] && cmd="$(resolve "$CFG_ISONET_CMD")" && usable isonet2 "$cmd" && isonet_ok "$cmd" ""; then
        WHERE[isonet2]="$cmd"
        NOTE[isonet2]="config uses a bare command; pinning [python, isonet.py] is safer (README)"
        return 0
    fi
    # pipeline.local's pair; then this prefix's own install and every conda
    # environment with IsoNet2 pip-installed (its bin/ has isonet.py), those
    # installed from the tested commit first -- an older IsoNet2 beta in
    # another environment shares the same version number
    env_bins
    local found=("$PREFIX/envs/isonet2/bin/python" "$PREFIX/IsoNet2/IsoNet/bin/isonet.py")
    for b in "${ENV_BINS[@]}"; do
        [ -f "$b/isonet.py" ] && found+=("$b/python" "$b/isonet.py")
    done
    local i at_ref=() other=()
    for ((i = 0; i < ${#found[@]}; i += 2)); do
        if isonet_at_ref "${found[i]}" "${found[i+1]}"; then at_ref+=("${found[@]:i:2}")
        else other+=("${found[@]:i:2}"); fi
    done
    local pairs=("${PIPELINE_ISONET_PYTHON:-}" "${PIPELINE_ISONET_SCRIPT:-}" "${at_ref[@]}" "${other[@]}")
    for ((i = 0; i < ${#pairs[@]}; i += 2)); do
        py="${pairs[i]}" sc="${pairs[i+1]}"
        [ -n "$py" ] && [ -f "$sc" ] && usable isonet2 "$py" || continue
        if isonet_ok "$py" "$sc"; then
            WHERE[isonet2]="$sc ($py)"; ISONET_PY="$py"; ISONET_SCRIPT="$sc"
            SET_ISONET_PY="$py"; SET_ISONET_SCRIPT="$sc"
            isonet_at_ref "$py" "$sc" || NOTE[isonet2]="not installed from the tested commit $ISONET2_REF"
            return 0
        fi
    done
    REASON="${REASON:-no IsoNet2 found}"
    return 1
}
isonet_at_ref() {  # isonet_at_ref <python> <isonet.py>: is that IsoNet2's git checkout at $ISONET2_REF?
    local src head want
    case "$2" in
        */IsoNet/bin/isonet.py) src="${2%/IsoNet/bin/isonet.py}" ;;   # run from a checkout
        *)  # pip-installed: the checkout it was installed from
            src="$(PYTHONNOUSERSITE=1 timeout 60 "$1" -I -c '
import importlib.metadata as m, json
u = json.loads(m.distribution("IsoNet2").read_text("direct_url.json") or "{}").get("url", "")
print(u[7:] if u.startswith("file://") else "")' 2>/dev/null)" ;;
    esac
    [ -n "$src" ] || return 1
    head="$(git -C "$src" rev-parse -q --verify HEAD 2>/dev/null)" &&
        want="$(git -C "$src" rev-parse -q --verify "$ISONET2_REF^{commit}" 2>/dev/null)" &&
        [ "$head" = "$want" ]
}
install_isonet2() {
    local src="$PREFIX/IsoNet2" env="$PREFIX/envs/isonet2"
    ensure_conda || return 1
    # leaves out what the pipeline doesn't use: the GUI app, the tutorial's
    # images and the preprint PDF (most of the download)
    git_checkout isonet2 "$ISONET2_REPO" "$src" "$ISONET2_REF" \
        '!/IsoApp/' '!/IsoNet/tutorial_figures/' '!/*.pdf' || return 1
    if [ -x "$env/bin/python" ]; then
        run_log isonet2 "$CONDA" env update -p "$env" -f "$src/isonet2_environment.yml"
    else
        run_log isonet2 "$CONDA" env create -p "$env" -f "$src/isonet2_environment.yml"
    fi || { fail_tail "conda environment from isonet2_environment.yml failed" isonet2; return 1; }
    run_log isonet2 "$env/bin/python" -m pip install "$src" || { fail_tail "pip install of IsoNet2 failed" isonet2; return 1; }
}

# ---------------------------------------------------------------- PyTom
pytom_ok() {  # pytom_ok <bin dir>
    local v
    [ -x "$1/pytom_match_template.py" ] && [ -x "$1/python" ] || { REASON="$1/pytom_match_template.py missing"; return 1; }
    timeout 300 "$1/pytom_match_template.py" --help >/dev/null 2>&1 || { REASON="pytom_match_template.py --help failed"; return 1; }
    v="$("$1/python" -c 'import importlib.metadata as m; print(m.version("pytom-match-pick"))' 2>/dev/null)"
    NOTE[pytom]="pytom-match-pick ${v:-?}"
    [[ "$v" == "${PYTOM_VERSION%.*}".* ]] || NOTE[pytom]+=" (tested with $PYTOM_VERSION)"
}
check_pytom() {
    local b envs=()
    env_bins
    for b in "${ENV_BINS[@]}"; do [ -f "$b/pytom_match_template.py" ] && envs+=("$b"); done
    for b in "$CFG_PYTOM_BIN" "${PIPELINE_PYTOM_BIN_DIR:-}" \
             "$(dirname "$(command -v pytom_match_template.py 2>/dev/null || echo /none/x)")" \
             "$PREFIX/envs/pytom_tm/bin" "${envs[@]}"; do
        [ -n "$b" ] && usable pytom "$b" && pytom_ok "$b" || continue
        WHERE[pytom]="$b"
        [ "$b" = "$CFG_PYTOM_BIN" ] || SET_PYTOM_BIN_DIR="$b"
        return 0
    done
    return 1
}
install_pytom() {
    local env="$PREFIX/envs/pytom_tm" drv cv=12.0
    ensure_conda || return 1
    # cupy's CUDA runtime must not be newer than the driver supports
    drv="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -n1)"
    [ "${drv:-12}" -lt 12 ] && cv=11.8
    if [ ! -x "$env/bin/python" ]; then
        run_log pytom "$CONDA" create -y -p "$env" --override-channels -c conda-forge \
            python=3.13 cupy "cuda-version=$cv" || { fail_tail "conda create failed" pytom; return 1; }
    fi
    run_log pytom "$env/bin/python" -m pip install "pytom-match-pick[plotting]==$PYTOM_VERSION" ||
        { fail_tail "pip install pytom-match-pick failed" pytom; return 1; }
}

# ---------------------------------------------------------------- AreTomo3 / AreTomo2
aretomo_ok() {  # aretomo_ok <binary>
    [ -f "$1" ] && [ -x "$1" ] || { REASON="$1 not executable"; return 1; }
    ! ldd "$1" 2>/dev/null | grep -q "not found" || { REASON="$1: $(ldd "$1" | grep 'not found' | head -n1 | xargs)"; return 1; }
    timeout 60 "$1" --help >/dev/null 2>&1 || { REASON="$1 --help failed"; return 1; }
}
check_aretomo() {  # check_aretomo <component> <config value> <binary name> <pipeline.local value>
    local p found
    mapfile -t found < <(search "$3" "[Aa]re[Tt]omo*/$3" "[Aa]re[Tt]omo*/bin/$3")
    for p in "$(resolve "$2")" "$(resolve "$4")" "$(resolve "$3")" "$PREFIX/$3/$3" "${found[@]}"; do
        usable "$1" "$p" && aretomo_ok "$p" || continue
        WHERE[$1]="$p"
        if [ "$p" != "$(resolve "$2")" ]; then
            if [ "$1" = aretomo3 ]; then SET_ARETOMO_BIN="$p"; else SET_ARETOMO2_BIN="$p"; fi
        fi
        return 0
    done
    return 1
}
find_cuda() {
    local c
    for c in "$CUDA_DIR" "${CUDA_HOME:-}" "${CUDA_PATH:-}" \
             "$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /none/x/x)")")" /usr/local/cuda; do
        [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && [ -e "$c/lib64/libcufft.so" ] && { readlink -f "$c"; return 0; }
    done
    return 1
}
install_aretomo() {  # install_aretomo <component> <name> <repo> <ref>
    local comp="$1" name="$2" src="$PREFIX/$2" cuda major mf cc
    [ -n "$PIPELINE_PY" ] || { REASON="needs the pipeline Python (to patch the makefile)"; return 1; }
    cuda="$(find_cuda)" || { REASON="no CUDA toolkit (bin/nvcc + lib64/libcufft.so) -- install one or pass --cuda DIR"; return 1; }
    [ -x /usr/bin/g++ ] || { REASON="no system C++ compiler /usr/bin/g++ (e.g. apt install build-essential)"; return 1; }
    command -v make >/dev/null || { REASON="make not found"; return 1; }
    git_checkout "$comp" "$3" "$src" "$4" || return 1
    major="$("$cuda/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
    if [ "$name" = AreTomo3 ]; then
        mf="makefile$major"; [ -f "$src/$mf" ] || mf=makefile12
    else
        mf=makefile11   # AreTomo2: CUDA 11+ (its README); "makefile" is for compute 5.x GPUs
    fi
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d .)"
    # The prebuilt LibSrc/Lib/*.a are not position-independent, so link a
    # non-PIE executable (the existing AreTomo3 build here needed the same
    # edit), and compile for this machine's GPU if the makefile doesn't.
    "$PIPELINE_PY" - "$src/$mf" "$cc" <<'PY' || { REASON="couldn't patch $mf"; return 1; }
import re, sys
path, cc = sys.argv[1], sys.argv[2]
text = open(path).read()
def nopie(m):
    line = m.group(0)
    if "no-pie" in line:
        return line
    flag = "-Xlinker -no-pie " if "NVCC" in line else "-no-pie "
    return line.replace("-m64 ", "-m64 " + flag, 1)
text = re.sub(r"(?m)(?<=^exe: \$\(OBJS\)\n)\t.*$", nopie, text)
if cc and f"code=sm_{cc}" not in text:
    text = re.sub(r"-gencode arch=compute_\d+,code=sm_\d+",
                  lambda m: f"-gencode arch=compute_{cc},code=sm_{cc} \\\n\t" + m.group(0), text, count=1)
open(path, "w").write(text)
PY
    run_log "$comp" env -i HOME="$HOME" LANG=C PATH="$cuda/bin:/usr/bin:/bin" \
        make -C "$src" -f "$mf" exe CUDAHOME="$cuda" -j"$JOBS" ||
        { fail_tail "build failed (CUDA $cuda, $mf)" "$comp"; return 1; }
}

# ---------------------------------------------------------------- run
check() {  # check <component>
    case "$1" in
        pipeline) check_pipeline ;;
        imod)     check_imod ;;
        isonet2)  check_isonet2 ;;
        pytom)    check_pytom ;;
        aretomo3) check_aretomo aretomo3 "$CFG_ARETOMO3" AreTomo3 "${PIPELINE_ARETOMO3_BIN:-}" ;;
        aretomo2) check_aretomo aretomo2 "$CFG_ARETOMO2" AreTomo2 "${PIPELINE_ARETOMO2_BIN:-}" ;;
    esac
}
install() {  # install <component>
    case "$1" in
        pipeline) install_pipeline ;;
        imod)     install_imod ;;
        isonet2)  install_isonet2 ;;
        pytom)    install_pytom ;;
        aretomo3) install_aretomo aretomo3 AreTomo3 "$ARETOMO3_REPO" "$ARETOMO3_REF" ;;
        aretomo2) install_aretomo aretomo2 AreTomo2 "$ARETOMO2_REPO" "$ARETOMO2_REF" ;;
    esac
}

say "Install prefix: $PREFIX"
say "Machine:        $HOST (settings in $LOCAL)"
[ -n "$CONFIG" ] && say "Config:         $CONFIG"
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 ||
    warn "no working NVIDIA driver (nvidia-smi) -- AreTomo, IsoNet2 and PyTom all need a GPU"
for t in git make; do command -v "$t" >/dev/null || warn "$t not found (needed to install from source)"; done
command -v curl >/dev/null || command -v wget >/dev/null || warn "neither curl nor wget found (needed for downloads)"

confirm() {
    [ "$YES" = 1 ] && return 0
    if [ -t 0 ]; then
        read -r -p "$1 [y/N] " a
        [[ "$a" == [yY]* ]]
    else
        say "$1 -- not a terminal, rerun with --yes to install."
        exit 1
    fi
}
install_one() {  # install_one <component>: install, then check it again
    say "== installing $1 (log: $LOGDIR/$1.log)"
    REASON=""
    if install "$1" && { NOTE[$1]=""; check "$1"; }; then STATUS[$1]="installed"
    else STATUS[$1]="FAILED"; NOTE[$1]="$REASON"; fi
}

# 1. The pipeline's own Python comes first (always checked): it reads and
#    writes the config and patches the AreTomo makefiles.
say "checking pipeline ..."
if check_pipeline; then STATUS[pipeline]="found"
else
    STATUS[pipeline]="missing"; NOTE[pipeline]="$REASON"
    if [ "$MODE" = install ]; then
        confirm "Install the pipeline's Python environment into $PREFIX/envs/cryoet_pipeline (needed to read the config)?" ||
            { say "Nothing installed."; exit 0; }
        install_one pipeline
    fi
fi
read_config
[ -n "$PIPELINE_PY" ] || [ -z "$CONFIG" ] ||
    warn "can't read $CONFIG without the pipeline Python -- its software paths weren't checked"

# 2. Everything else.
PLAN=()
for c in "${COMPONENTS[@]:1}"; do
    wanted "$c" || { STATUS[$c]="skipped"; continue; }
    say "checking $c ..."
    REASON=""; NOTE[$c]=""
    if check "$c"; then STATUS[$c]="found"
    else STATUS[$c]="missing"; NOTE[$c]="$REASON"; PLAN+=("$c"); fi
done
if [ "$MODE" = install ] && [ ${#PLAN[@]} -gt 0 ]; then
    # rough cost of each, measured on the test machine (fast network, 128 cores)
    declare -A COST=([imod]="~0.6 GB, ~1 min" [isonet2]="~10 GB, ~15 min (PyTorch + CUDA)"
                     [pytom]="~3 GB, ~4 min" [aretomo3]="build, 1-10 min" [aretomo2]="build, 1-5 min")
    say ""
    say "Missing -- to install:"
    for c in "${PLAN[@]}"; do say "   $c  (${COST[$c]})"; done
    confirm "Install them into $PREFIX?" || { say "Nothing more installed."; PLAN=(); }
    for c in "${PLAN[@]}"; do install_one "$c"; done
fi

# ---------------------------------------------------------------- pipeline.local, config, launchers
pick() {  # pick <component> <value found now> <value pipeline.local had>
    case "${STATUS[$1]:-skipped}" in
        found|installed) printf '%s' "$2" ;;
        skipped) printf '%s' "$3" ;;   # not checked this time (--only/--skip): keep it
    esac
}
write_local() {
    # This machine's block in pipeline.local, replacing the one an earlier
    # run wrote: blocks of other machines sharing this checkout (e.g. on a
    # network drive) stay as they are.
    local begin="# >>> install.sh: $HOST >>>" end="# <<< install.sh: $HOST <<<"
    local a3 a2 d add="" block tmp="$LOCAL.tmp.$$" var val
    a3="$(pick aretomo3 "${WHERE[aretomo3]:-}" "${PIPELINE_ARETOMO3_BIN:-}")"
    a2="$(pick aretomo2 "${WHERE[aretomo2]:-}" "${PIPELINE_ARETOMO2_BIN:-}")"
    # for configs that call AreTomo by bare name; not checked against this
    # account's PATH -- the others using ./pipeline don't have its ~/.bashrc
    for d in "$a3" "$a2"; do
        [ -n "$d" ] || continue
        d="$(dirname "$d")"
        [[ ":/usr/local/bin:/usr/bin:/bin$add:" == *":$d:"* ]] || add="$add:$d"
    done
    block="$begin
# written by install.sh on $(date '+%Y-%m-%d %H:%M'); rerun it to update
if [ \"\$(hostname -s)\" = $(printf %q "$HOST") ]; then
    PIPELINE_PYTHON=$(printf %q "$PIPELINE_PY")"
    [ -n "$add" ] && block+="
    PATH=\"\$PATH$add\""
    block+="
    # this machine's software: \"./pipeline new\" puts these into the configs it makes"
    for var in PIPELINE_ARETOMO3_BIN PIPELINE_ARETOMO2_BIN PIPELINE_ISONET_PYTHON PIPELINE_ISONET_SCRIPT \
               PIPELINE_IMOD_BIN_DIR PIPELINE_PYTOM_BIN_DIR; do
        case "$var" in
            PIPELINE_ARETOMO3_BIN)  val="$a3" ;;
            PIPELINE_ARETOMO2_BIN)  val="$a2" ;;
            PIPELINE_ISONET_PYTHON) val="$(pick isonet2 "$ISONET_PY" "${PIPELINE_ISONET_PYTHON:-}")" ;;
            PIPELINE_ISONET_SCRIPT) val="$(pick isonet2 "$ISONET_SCRIPT" "${PIPELINE_ISONET_SCRIPT:-}")" ;;
            PIPELINE_IMOD_BIN_DIR)  val="$(pick imod "${WHERE[imod]:-}" "${PIPELINE_IMOD_BIN_DIR:-}")" ;;
            PIPELINE_PYTOM_BIN_DIR) val="$(pick pytom "${WHERE[pytom]:-}" "${PIPELINE_PYTOM_BIN_DIR:-}")" ;;
        esac
        [ -n "$val" ] && block+="
    export $var=$(printf %q "$val")"
    done
    block+="
fi
$end"
    if {
        if [ -f "$LOCAL" ]; then
            awk -v b="$begin" -v e="$end" '$0 == b {skip = 1} !skip {print} $0 == e {skip = 0}' "$LOCAL"
        else
            printf '%s\n' "# Machine settings for ./pipeline (not in git; see the comment at the top of" \
                          "# ./pipeline). One block per machine using this checkout, chosen by hostname." ""
        fi
        printf '%s\n' "$block"
    } > "$tmp" 2>/dev/null; then
        [ -f "$LOCAL" ] && chmod --reference="$LOCAL" "$tmp"
        chmod a+r "$tmp"
        mv "$tmp" "$LOCAL" && { say "pipeline.local: wrote the settings for $HOST"; return 0; }
    fi
    rm -f "$tmp"
    warn "couldn't write $LOCAL -- ask its owner to add (or replace) this block:"
    printf '%s\n' "$block"
}

if [ "$MODE" = install ] && [ -n "$PIPELINE_PY" ]; then
    say ""
    write_local
fi
if [ "$MODE" = install ] && [ -n "$PIPELINE_PY" ] && [ -n "$CONFIG" ]; then
    if [ ! -f "$CONFIG" ]; then
        cp "$REPO/pipeline_config.example.yaml" "$CONFIG" && say "Created $CONFIG from pipeline_config.example.yaml"
    fi
    SET_ARETOMO_BIN="$SET_ARETOMO_BIN" SET_ARETOMO2_BIN="$SET_ARETOMO2_BIN" \
    SET_IMOD_BIN_DIR="$SET_IMOD_BIN_DIR" SET_PYTOM_BIN_DIR="$SET_PYTOM_BIN_DIR" \
    SET_ISONET_PY="$SET_ISONET_PY" SET_ISONET_SCRIPT="$SET_ISONET_SCRIPT" \
    "$PIPELINE_PY" - "$CONFIG" "$REPO" <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from config_editor import set_values
e = os.environ
updates = {k: e[v] for k, v in (("aretomo_bin", "SET_ARETOMO_BIN"), ("aretomo2_bin", "SET_ARETOMO2_BIN"),
                                 ("imod.bin_dir", "SET_IMOD_BIN_DIR"), ("pytom.bin_dir", "SET_PYTOM_BIN_DIR")) if e[v]}
if e["SET_ISONET_PY"]:
    updates["isonet_bin"] = [e["SET_ISONET_PY"], e["SET_ISONET_SCRIPT"]]
missing = set_values(Path(sys.argv[1]), updates)
for k, v in updates.items():
    if k not in missing:
        print(f"Config: set {k}: {v}")
if missing:
    print(f"\nAdd these lines to {sys.argv[1]} yourself (their sections are commented out there):")
    for k in missing:
        sec, _, key = k.rpartition(".")
        print(f"  {sec + ':' if sec else ''}\n    {key}: {updates[k]!r}" if sec else f"  {key}: {updates[k]!r}")
PY
fi
if [ "$MODE" = install ] && [ -n "$PIPELINE_PY" ]; then
    mkdir -p "$PREFIX/bin"
    for pair in "cryoet-pipeline:run_pipeline.py" "cryoet-config-editor:config_editor.py"; do
        printf '#!/usr/bin/env bash\n# written by install.sh: runs with the Python it checked, ignoring ~/.local\nexport PYTHONNOUSERSITE=1\nexec %q %q "$@"\n' \
            "$PIPELINE_PY" "$REPO/${pair#*:}" > "$PREFIX/bin/${pair%%:*}"
        chmod +x "$PREFIX/bin/${pair%%:*}"
    done
fi

# ---------------------------------------------------------------- summary
say ""
printf '%-9s %-9s %s\n' component status "location / note"
rc=0
for c in "${COMPONENTS[@]}"; do
    printf '%-9s %-9s %s\n' "$c" "${STATUS[$c]:-skipped}" "${WHERE[$c]:-}${NOTE[$c]:+${WHERE[$c]:+  -- }${NOTE[$c]}}"
    case "${STATUS[$c]:-}" in missing|FAILED) rc=1 ;; esac
done
if [ "$MODE" = install ] && [ -n "$PIPELINE_PY" ]; then
    say ""
    say "Next:  ./pipeline new MyData            # a numbered config with this machine's software"
    say "       ./pipeline edit N                # put in your data, output folders, GPUs"
    say "       ./pipeline run N --dry-run       # then without --dry-run"
    say "A config file of your own:  $PREFIX/bin/cryoet-pipeline --config <file>"
fi
exit $rc
