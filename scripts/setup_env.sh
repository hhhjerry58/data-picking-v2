#!/usr/bin/env bash
set -euo pipefail
data_picking_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$data_picking_repo"
data_picking_python="${PYTHON_BIN:-python3.12}"
"$data_picking_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing checkpoints require Python 3.12"'
if [ ! -x .venv/bin/python ]; then
  "$data_picking_python" -m venv .venv
fi
source scripts/env.sh
.venv/bin/python -m pip install --upgrade pip
if [ "${DATA_PICKING_SKIP_TORCH_INSTALL:-0}" != 1 ]; then
  .venv/bin/python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
fi
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python - <<'PY'
import json,sys,torch,transformers,numpy
from pathlib import Path
expected=json.loads(Path('runtime_versions.json').read_text())
assert f'{sys.version_info.major}.{sys.version_info.minor}' == expected['python_major_minor']
assert torch.__version__ == expected['torch']
assert torch.version.git_version == expected['torch_git_version']
assert transformers.__version__ == expected['transformers']
assert numpy.__version__ == expected['numpy']
assert torch.version.cuda == expected['cuda_build']
from picking.domain_retrieval.config import load_config
from picking.domain_retrieval.training import TokenStream
load_config('configs/domain_retrieval.yaml')
print('Domain retrieval environment ready; existing checkpoints remain version-pinned.')
PY
