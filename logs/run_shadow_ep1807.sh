#!/usr/bin/env bash
# T5 shadow 루프 — ep1807 전 구간. **로컬 쪽**이다 (서버는 따로 띄워 둔다).
#
# shadow = AG3S · cuRobo · SQP 를 전부 돌리되 **수정된 청크를 로봇에 보내지 않는** 실행.
# 로봇은 정책의 원본 청크(`actions_reference`)를 날리고, refined 와 판정은 기록만 된다.
# 그래야 "수정이 여유거리를 나쁘게 만드는가" 를 로봇을 움직이기 전에 볼 수 있다.
#
# **서버도 `--shadow` 로 떠 있어야 한다.** 한쪽만 켜면 로봇이 움직이기 전에 RuntimeError 로
# 죽는다 (양방향 검사, `client.py` 생성자 + 첫 왕복).
#
#   PORT=8000 OUT=outputs/live_test/20260925_t5shadow bash pi05_TO_hybrid/logs/run_shadow_ep1807.sh
#
# ep1807 인 이유: **과제가 물리적으로 완결되는 기록**이다 (T2 에서 확정 — apple 이 240.6 mm
# 들려 339 -> 71 mm 로 옮겨진다). ep1800 은 92.1 mm 들렸다 제자리로 돌아와 운반이 없어
# 뒤 절반을 잴 수 없었다.
#
# 600 제어 스텝 = 75 청크가 맞는 조건이다 (사용자 판정 2026-09-25). 늘리지 않는다.
set -euo pipefail

cd /mnt/dev/work

export MUJOCO_GL=osmesa
export PYTHONPATH=/mnt/dev/work
# 기록의 `model_xml` 은 녹화한 PC 의 절대경로다. **심볼릭 링크를 만들지 않는다** —
# 컨테이너 재시작에 날아가 재생이 전부 죽은 적이 있다.
export AG3S_ASSET_ROOT=/mnt/dev/work/pi05_TO_hybrid

out="${OUT:-outputs/live_test/20260925_t5shadow}/shadow_ep1807"
mkdir -p "$out"

exec .venv-openpi-live/bin/python -u pi05_TO_hybrid/rby1_bringup/pi05_infer.py \
  --model rby1_randomized_pick_place_16d \
  --remote "localhost:${PORT:-8000}" \
  --safe-remote --safe-shadow \
  --safe-timeout 120 \
  --episode-index 1807 \
  --max-steps 600 \
  --headless --start-delay 0 --speed 0 \
  --safe-phase approach --safe-manipulators left \
  --record-frames "$out" \
  "$@"
