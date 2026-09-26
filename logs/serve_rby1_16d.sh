#!/usr/bin/env bash
set -euo pipefail
workspace=/mnt/dev/work/pi05_TO_hybrid
cd "$workspace/openpi"
export OPENPI_DATA_HOME="$workspace/cache/openpi"
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
exec .venv/bin/python scripts/serve_policy.py \
  --port "${PORT:-8000}" \
  --default-prompt "Put the orange in the basket." \
  policy:checkpoint \
  --policy.config pi05_rby1_randomized_pick_place_16d_lora \
  --policy.dir "$workspace/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999"
