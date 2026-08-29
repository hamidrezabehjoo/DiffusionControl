#!/usr/bin/env bash
# ADK basin-escape experiment: stock Boltz-2 vs PMP noise-controlled Boltz-2.
#
# Prerequisites:
#   1. Python env with GPU:  pip install "boltz[cuda]==2.2.1"
#   2. Apply the controlled-sampler patch (once per environment):
#        BOLTZ_PKG=$(python -c "import boltz, os; print(os.path.dirname(boltz.__file__))")
#        cp diffusionv2.py "$BOLTZ_PKG/model/modules/diffusionv2.py"
#   3. Model weights download automatically on first run (~6 GB).
#
# The patch is controlled by two environment variables:
#   NU_ALPHA  noise gain (default 1.0 = stock, bit-exact)
#   NU_TW     late-reverse window [0, t_w] in normalized time (default 0.2)
#
# The model weights + CCD (~8 GB) download to the cache dir, and Triton's JIT
# kernel cache plus other tool caches default to $HOME. If $HOME has a tight
# disk quota, point ONE root at a big filesystem and everything follows:
#   export BIGCACHE=/scratch/$USER/caches
# (individual variables below can still be overridden separately.)

set -euo pipefail
cd "$(dirname "$0")"

BIGCACHE="${BIGCACHE:-$HOME}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$BIGCACHE/boltz_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BIGCACHE/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$BIGCACHE/torchinductor_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$BIGCACHE/cuda_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BIGCACHE/xdg_cache}"
mkdir -p "$BOLTZ_CACHE" "$TRITON_CACHE_DIR"

CACHE="$BOLTZ_CACHE"

# Preflight: caches need ~10 GB (first run), outputs ~1 GB. Fail early with a
# clear message instead of a deep OSError mid-prediction. Note: a path under
# $HOME (e.g. ~/scratch) shares the home quota unless it is a symlink out.
check_space() {  # $1 = path, $2 = required GB, $3 = label
    local target="$1" req_gb="$2" label="$3"
    [ -d "$target" ] || target=$(dirname "$target")
    local avail_gb
    avail_gb=$(( $(df -k --output=avail "$target" | tail -1) / 1048576 ))
    echo "preflight: $label -> $(df -h "$target" | tail -1 | awk '{print $4" free on "$6" ("$1")"}')"
    if [ "$avail_gb" -lt "$req_gb" ]; then
        echo "ERROR: $label has ${avail_gb} GB free, need ~${req_gb} GB." >&2
        echo "Point BIGCACHE at a larger filesystem (df -h to find one;" >&2
        echo "beware: ~/scratch is still under your home quota)." >&2
        exit 1
    fi
}
check_space "$BOLTZ_CACHE" 10 "BIGCACHE ($BOLTZ_CACHE)"
check_space "$(pwd)" 1 "output dir ($(pwd))"
N_STOCK=${N_STOCK:-100}     # stock ensemble size (collapse statistics)
N_CTRL=${N_CTRL:-20}        # controlled ensemble size (escape fraction)
STEPS=${STEPS:-200}
SEED=${SEED:-42}

# ---------------------------------------------------------------- stock ----
# NU_ALPHA unset -> stock sampler, untouched code path.
boltz predict adk.yaml \
    --use_msa_server --use_potentials \
    --diffusion_samples "$N_STOCK" --sampling_steps "$STEPS" \
    --output_format pdb --seed "$SEED" \
    --cache "$CACHE" \
    --out_dir results_stock --override

python3 eval_adk.py --pred_dir results_stock/boltz_results_adk/predictions/adk \
    --tag stock --out eval_stock.csv

# ------------------------------------------------------------ controlled ----
# Paper setting: alpha = 2.5 on the window [0, 0.2].
NU_ALPHA=2.5 NU_TW=0.2 \
boltz predict adk.yaml \
    --use_msa_server --use_potentials \
    --diffusion_samples "$N_CTRL" --sampling_steps "$STEPS" \
    --output_format pdb --seed "$SEED" \
    --cache "$CACHE" \
    --out_dir results_a2.5_tw0.2 --override

python3 eval_adk.py --pred_dir results_a2.5_tw0.2/boltz_results_adk/predictions/adk \
    --tag alpha2.5_tw0.2 --out eval_a2.5_tw0.2.csv

# Optional second setting reported in the paper: alpha = 3 on [0, 0.3].
# NU_ALPHA=3.0 NU_TW=0.3 \
# boltz predict adk.yaml \
#     --use_msa_server --use_potentials \
#     --diffusion_samples "$N_CTRL" --sampling_steps "$STEPS" \
#     --output_format pdb --seed "$SEED" \
#     --cache "$CACHE" \
#     --out_dir results_a3_tw0.3 --override
# python3 eval_adk.py --pred_dir results_a3_tw0.3/boltz_results_adk/predictions/adk \
#     --tag alpha3_tw0.3 --out eval_a3_tw0.3.csv
