#!/usr/bin/env bash
# T6 실행 모드 — 서버가 다듬은 chunk 를 로컬이 **실제로 실행**한다.
#
# shadow 와 다른 점은 둘뿐이다:
#   - 서버를 `--shadow` **없이** 띄운다 (응답에 `actions_reference` 가 안 실린다)
#   - 로컬에 `--safe-shadow` 를 **주지 않는다**
# 그러면 `pi05_infer.py:1557` 의 hold gate 가 살아나 **unsafe 판정이면 로봇이 멈춘다.**
# shadow 에서는 7 chunk 가 unsafe 였으므로 여기서는 그 7 번 멈출 것으로 예상된다 —
# 예상이지 사실이 아니다.
#
#   PORT=8000 OUT=outputs/live_test/20260926_t6 bash pi05_TO_hybrid/logs/run_execute_ep1807.sh
set -euo pipefail
cd /mnt/dev/work
export MUJOCO_GL=osmesa
export PYTHONPATH=/mnt/dev/work
export AG3S_ASSET_ROOT=/mnt/dev/work/pi05_TO_hybrid
out="${OUT:-outputs/live_test/20260926_t6}/execute_ep1807"
mkdir -p "$out"
exec .venv-openpi-live/bin/python -u pi05_TO_hybrid/rby1_bringup/pi05_infer.py \
  --model rby1_randomized_pick_place_16d \
  --remote "localhost:${PORT:-8000}" \
  --safe-remote \
  --safe-timeout 120 \
  --episode-index 1807 \
  --max-steps 600 \
  --headless --start-delay 0 --speed 0 \
  --safe-phase approach --safe-manipulators left \
  --record-frames "$out" \
  "$@"
