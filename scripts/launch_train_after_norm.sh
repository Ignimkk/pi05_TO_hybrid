#!/usr/bin/env bash
set -euo pipefail

project=/mnt/dev/work/pi05_TO_hybrid
norm_session=rby1_transport_norm
train_session=rby1_pi05_lora_30k
norm_log="$project/logs/rby1_transport_norm_20260812.log"
guard_log="$project/logs/rby1_training_guard_20260812.log"
norm_file="$project/assets/pi05_rby1_lora/local/rby1_dataset_v1/norm_stats.json"
train_log="$project/logs/pi05_rby1_lora_30k_20260812.log"
experiment=rby1_transport_14d_30k_20260812
checkpoint_dir="$project/checkpoints/pi05_rby1_lora/$experiment"

exec >>"$guard_log" 2>&1
echo "$(date -Is) waiting for $norm_session"
while tmux has-session -t "$norm_session" 2>/dev/null; do
  sleep 30
done

if ! grep -q 'Generating train split: 609094 examples' "$norm_log"; then
  echo "$(date -Is) ERROR: expected 609094-example train split was not observed"
  exit 1
fi
if ! grep -q 'Writing stats to:' "$norm_log"; then
  echo "$(date -Is) ERROR: norm computation did not report successful output"
  exit 1
fi
if [[ ! -s "$norm_file" ]]; then
  echo "$(date -Is) ERROR: missing or empty $norm_file"
  exit 1
fi
"$project/openpi/.venv/bin/python" -m json.tool "$norm_file" >/dev/null
if [[ -e "$checkpoint_dir" ]]; then
  echo "$(date -Is) ERROR: checkpoint destination already exists: $checkpoint_dir"
  exit 1
fi
if tmux has-session -t "$train_session" 2>/dev/null; then
  echo "$(date -Is) ERROR: training tmux session already exists: $train_session"
  exit 1
fi

echo "$(date -Is) norm stats validated; starting $train_session"
tmux new-session -d -s "$train_session" \
  "cd '$project' && export HF_DATASETS_CACHE='$project/.cache/hf_datasets_rby1_corrected_20260812' HF_LEROBOT_HOME='$project/lerobot_transport' WANDB_MODE=disabled && exec openpi/.venv/bin/python openpi/scripts/train.py pi05_rby1_lora --exp-name '$experiment' --assets-base-dir '$project/assets' --checkpoint-base-dir '$project/checkpoints' --num-train-steps 30000 --no-wandb-enabled 2>&1 | tee '$train_log'"
echo "$(date -Is) started $train_session; checkpoints: $checkpoint_dir"
