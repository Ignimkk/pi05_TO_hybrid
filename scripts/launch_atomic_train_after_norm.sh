#!/usr/bin/env bash
set -euo pipefail

project=/mnt/dev/work/pi05_TO_hybrid
dataset="$project/data/rby1_atomic_basket_14d_v2"
lerobot_home="$project/lerobot_atomic_20260825"
cache="$project/.cache/hf_datasets_rby1_atomic_20260825"
assets_base="$project/assets"
norm_session=rby1_atomic_norm
train_session=rby1_atomic_lora_30k
norm_log="$project/logs/rby1_atomic_norm_20260825.log"
guard_log="$project/logs/rby1_atomic_guard_20260825.log"
train_log="$project/logs/rby1_atomic_lora_30k_20260825.log"
experiment=rby1_atomic_basket_14d_v2_30k_20260825
norm_file="$assets_base/pi05_rby1_atomic_lora/local/rby1_atomic_basket_14d_v2/norm_stats.json"
checkpoint_dir="$project/checkpoints/pi05_rby1_atomic_lora/$experiment"
repo_path="$lerobot_home/local/rby1_atomic_basket_14d_v2"

exec >>"$guard_log" 2>&1
if [[ ! -e "$repo_path" ]]; then
  echo "$(date -Is) ERROR: missing LeRobot repo path: $repo_path"
  exit 1
fi
if [[ "$(readlink -f "$repo_path")" != "$(readlink -f "$dataset")" ]]; then
  echo "$(date -Is) ERROR: LeRobot repo does not resolve to dataset"
  echo "  repo: $(readlink -f "$repo_path")"
  echo "  data: $(readlink -f "$dataset")"
  exit 1
fi
echo "$(date -Is) waiting for $norm_session"
while tmux has-session -t "$norm_session" 2>/dev/null; do
  sleep 30
done

if ! grep -q 'Writing stats to:' "$norm_log"; then
  echo "$(date -Is) ERROR: norm computation did not report successful output"
  exit 1
fi
"$project/openpi/.venv/bin/python" -c \
  'from openpi.training.config import get_config; e=get_config("pi05_rby1_atomic_lora").data.base_config.episodes; assert tuple(e)==tuple(range(1591)); print("OpenPI train episode selection: 0:1591 OK")'
if [[ ! -s "$norm_file" ]]; then
  echo "$(date -Is) ERROR: missing or empty $norm_file"
  exit 1
fi
"$project/openpi/.venv/bin/python" -m json.tool "$norm_file" >/dev/null
"$project/openpi/.venv/bin/python" "$project/scripts/preflight_atomic_finetune.py" \
  --dataset "$dataset" --require-stats --check-video-frames
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
  "cd '$project' && export HF_DATASETS_CACHE='$cache' HF_LEROBOT_HOME='$lerobot_home' WANDB_MODE=disabled XLA_FLAGS='--xla_gpu_enable_command_buffer=' XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 && exec openpi/.venv/bin/python openpi/scripts/train.py pi05_rby1_atomic_lora --exp-name '$experiment' --assets-base-dir '$assets_base' --checkpoint-base-dir '$project/checkpoints' --num-train-steps 30000 --no-wandb-enabled 2>&1 | tee '$train_log'"
echo "$(date -Is) started $train_session; checkpoints: $checkpoint_dir"
