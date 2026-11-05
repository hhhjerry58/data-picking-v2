#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
base_python="${PYTHON_BIN:-python3.12}"
"$base_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing checkpoints require Python 3.12"'
if [ ! -x .venv/bin/python ]; then
  "$base_python" -m venv .venv
fi
source scripts/env.sh
.venv/bin/python -m pip install --upgrade pip
if [ "${DATA_PICKING_SKIP_TORCH_INSTALL:-0}" != 1 ]; then
  .venv/bin/python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
fi
.venv/bin/python -m pip install -r requirements-dev.txt
bash scripts/build_fasttext.sh
.venv/bin/python - <<'PY'
import json,sys,torch,transformers,numpy
from pathlib import Path
expected=json.loads(Path('runtime_versions.json').read_text())
assert f'{sys.version_info.major}.{sys.version_info.minor}' == expected['python_major_minor']
assert torch.__version__ == expected['torch']
assert torch.version.git_version == expected['torch_git_version']
assert transformers.__version__ == expected['transformers']
assert numpy.__version__ == expected['numpy']
from picking import _fasttext_native
print('Independent environment and native fastText extension ready.')
PY
