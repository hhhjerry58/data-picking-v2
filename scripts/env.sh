#!/usr/bin/env bash
data_picking_runtime="${DATA_PICKING_RUNTIME_ROOT:-/dev/shm/data-picking-v2-runtime}"
data_picking_cache="${DATA_PICKING_CACHE_ROOT:-/mnt/tianzhu_hot/data-picking-v2/cache}"
export DATA_PICKING_RUNTIME_ROOT="$data_picking_runtime"
export HF_HOME="$data_picking_cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME="$data_picking_cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$data_picking_cache/torchinductor"
export TRITON_CACHE_DIR="$data_picking_cache/triton"
export CUDA_CACHE_PATH="$data_picking_cache/cuda"
export MPLCONFIGDIR="$data_picking_cache/matplotlib"
export XDG_CACHE_HOME="$data_picking_cache/xdg"
export PIP_CACHE_DIR="$data_picking_cache/pip"
export TMPDIR="$data_picking_runtime/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TMPDIR"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
