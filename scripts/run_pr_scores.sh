#!/usr/bin/env bash
# Resume phase one only. There is intentionally no select/train/evaluate stage.
set -Eeuo pipefail
pr_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pr_repo"
: "${DATA_PICKING_GPU_IDS:?Explicitly set the allowed GPU IDs}"
pr_config="${1:-configs/pr_selection.yaml}"
pr_run="$({ .venv-pr/bin/python - "$pr_config" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,'src')
from picking.pr_selection.config import load_config, verify_artifact_mount, run_dir
cfg=load_config(sys.argv[1])
verify_artifact_mount(cfg)
print(run_dir(cfg))
PY
} )"
mkdir -p "$pr_run/logs"
exec 9>"/tmp/pr-scoring-launch-${pr_run##*/}.lock"
flock -n 9
pr_stage=prepare
pr_status() {
  .venv-pr/bin/python - "$pr_run" "$$" "$pr_stage" "$1" "${2:-0}" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,'src')
from picking.pr_selection.config import atomic_json
atomic_json(Path(sys.argv[1])/'pipeline_status.json', {
    'launcher_pid':int(sys.argv[2]), 'stage':sys.argv[3], 'status':sys.argv[4],
    'exit_code':int(sys.argv[5]), 'updated_at_utc':datetime.now(timezone.utc).isoformat()})
PY
}
trap 'pr_exit=$?; pr_status failed "$pr_exit" || true; exit "$pr_exit"' ERR
pr_status running
.venv-pr/bin/python -u scripts/pr_selection.py prepare --config "$pr_config" > "$pr_run/logs/prepare.log" 2>&1
pr_frozen="$pr_run/effective_config.json"
# probe/score also verify the preflight identity, so an outdated marker cannot
# authorize a different implementation or configuration.
pr_imported="$({ .venv-pr/bin/python - "$pr_frozen" <<'PYIMPORT'
import json,sys
print('yes' if json.load(open(sys.argv[1])).get('probe_from_run') else 'no')
PYIMPORT
} )"
if [[ "$pr_imported" == no && ! -f "$pr_run/PREFLIGHT_PASSED.json" ]]; then
  pr_stage=preflight
  pr_status running
  .venv-pr/bin/python -u scripts/pr_selection.py preflight --config "$pr_frozen" > "$pr_run/logs/preflight.log" 2>&1
fi
for pr_stage in probe score analyze; do
  pr_status running
  .venv-pr/bin/python -u scripts/pr_selection.py "$pr_stage" --config "$pr_frozen" > "$pr_run/logs/$pr_stage.log" 2>&1
done
pr_status completed
