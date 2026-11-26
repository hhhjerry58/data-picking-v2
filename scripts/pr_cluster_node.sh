#!/usr/bin/env bash
set -euo pipefail
pr_bundle="$1"
pr_campaign="$2"
pr_node="$3"
pr_nodes="$4"
pr_mode="${5:-score}"
test -f "$pr_bundle/READY.json"
case "$pr_mode" in score|validate) ;; *) exit 2 ;; esac
pr_scratch="$(python3 - <<'PY'
import shutil
print('/dev/shm' if shutil.disk_usage('/dev/shm').free>180*1024**3 else '/tmp')
PY
)"
pr_local="$pr_scratch/pr-scoring-${pr_campaign}-${pr_node}"
mkdir -p "$pr_local"
cp "$pr_bundle/runtime.tar.gz" "$pr_local/runtime.tar.gz"
cp "$pr_bundle/code.tar.gz" "$pr_local/code.tar.gz"
(
  cd "$pr_local"
  sha256sum -c "$pr_bundle/archives.sha256"
  mkdir -p runtime code
  tar -xzf runtime.tar.gz -C runtime
  tar -xzf code.tar.gz -C code
  rm runtime.tar.gz code.tar.gz
)
export PYTHONPATH="$pr_local/runtime/overlay:$pr_local/runtime/base:$pr_local/code/src"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export DATA_PICKING_GPU_IDS=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pr_python="$pr_local/runtime/python/bin/python3.12"
"$pr_python" - <<'PY'
import torch,transformers,causal_conv1d
assert torch.__version__=='2.11.0+cu128'
assert transformers.__version__=='5.24.1'
assert torch.cuda.device_count()==8
assert all('A100' in torch.cuda.get_device_name(i) and torch.cuda.get_device_properties(i).total_memory>75*1024**3 for i in range(8))
print('A100_RUNTIME_VALIDATED',flush=True)
PY
pr_options=()
if [[ "$pr_mode" == validate ]]; then pr_options+=(--validate-only); fi
"$pr_python" -u -m picking.pr_parallel.cluster node \
  --bundle "$pr_bundle" --storage /mnt/tianzhu/data-picking-v2 \
  --local "$pr_local/assets" \
  --output "/mnt/tianzhu/data-picking-v2/runs/$pr_campaign/node-$(printf '%02d' "$pr_node")" \
  --node-id "$pr_node" --node-count "$pr_nodes" "${pr_options[@]}"
