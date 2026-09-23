#!/usr/bin/env bash
# One-click full pipeline (tiny smoke config by default; edit env vars for real runs).
set -euo pipefail
cd "$(dirname "$0")"

STEPS="${STEPS:-300}"          # optimizer steps per stage
VAL_INTERVAL="${VAL_INTERVAL:-100}"

python prepare_data.py --offline
python train_stage1.py --max-steps "$STEPS" --epochs 100 \
    --noise-std 0.1 --num-workers 0
python train_stage2.py --stage1-checkpoint checkpoints/stage1/best.pt \
    --max-steps "$STEPS" --epochs 100 --num-workers 0 --core transformer

python generate.py --checkpoint checkpoints/stage2/best.pt --prompt "小明今天去了学校"
python evaluate.py --checkpoint checkpoints/stage2/best.pt --max-probe-items 200
python inspect_latent.py --checkpoint checkpoints/stage1/best.pt --max-items 128
python demo.py --once
echo "ALL DONE ✔"
