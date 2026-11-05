# Project context

- This repository runs mathematics data-selection experiments using fixed 1.5B reference / math-SFT losses, two fastText classifiers, and paired 160M pretraining runs.
- Use Chinese for project discussion and review documents.
- Read `README.md` and `docs/DEVNODE.md` before continuing on a different node. Heavy data, weights and checkpoints live under `/mnt/tianzhu_hot/data-picking-v2`.
- Keep the numerical training file and training configuration compatible with the recorded checkpoint fingerprints. Do not bypass identity checks or overwrite the reviewed result snapshots.
- Choose available GPUs explicitly with `DATA_PICKING_GPU_IDS`; do not stop other jobs. Only one node may write a given run at a time.

# Training experiment preferences

- When creating a new training experiment, verify that checkpoints save optimizer state and automatic resume is enabled, unless the user explicitly requests otherwise.
- In `/home/v-guanhechen/rl`, experiments using `scripts/train-sopd-sing/train.sh` should explicitly set `--save_optimizer True` and `--auto_resume True`, with a valid persistent `--autosync_dir`. Check the effective configuration before launching or handing off the experiment.
