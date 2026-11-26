#!/usr/bin/env bash
set -euo pipefail
pr_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pr_repo"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-8}"
test -x .venv/bin/python
if [[ ! -x .venv-pr/bin/python ]]; then
  .venv/bin/python -m venv .venv-pr
fi
# Read shared Torch/analysis wheels without changing the historical environment.
.venv/bin/python - <<'PY'
from pathlib import Path
import sysconfig
base = Path(sysconfig.get_paths()['purelib']).resolve()
overlay = Path('.venv-pr/lib/python3.12/site-packages')
overlay.joinpath('base_environment.pth').write_text(str(base) + '\n')
PY
.venv-pr/bin/python -m pip install -r requirements-pr.txt --no-build-isolation
.venv-pr/bin/python - <<'PY'
import torch, transformers
assert torch.__version__ == '2.11.0+cu128'
from transformers import Qwen3_5ForCausalLM
print('PR environment:', torch.__version__, transformers.__version__)
PY
