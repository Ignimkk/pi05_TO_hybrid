#!/usr/bin/env bash
# serve_safe.py — 정책(π0.5) + AG3S + TO 를 **한 프로세스, 서버에서** 띄운다.
#
# 이 파일이 있는 이유: T0 를 재현하려 했을 때 서버를 어떻게 띄웠는지가 어디에도 없었다.
# 기록의 manifest `extra.argv` 에는 **클라이언트 인자만** 있고 (`--remote localhost:8000
# --safe-remote ...`), 서버 쪽 config·backend·복셀은 한 줄도 없다. 추측으로 재구성하면
# 그 순간 조건이 갈라진다. **서버를 띄우는 정본은 이 스크립트다.**
#
# 옆에 있는 `serve_rby1_16d.sh` 는 **bare policy** (안전 계층 없음) 다. 둘은 다른
# 프로그램이다 — 그쪽은 `serve_policy.py`, 이쪽은 `serve_safe.py` 로 AG3S+TO 를 얹는다.
#
#   PORT=8000 bash pi05_TO_hybrid/logs/serve_safe_rby1_16d.sh
#
# 아키텍처 (사용자 판정 2026-09-25): **AG3S·cuRobo·TO 는 서버에 둔다.** attention 과
# cuRobo 가 GPU 를 요구하기 때문이다. 로컬은 `pi05_infer.py --safe-remote` 로 관측과
# 프롬프트를 보내고 action 만 받는다. 물려받은 "T5·T6 에서 클라이언트 in-process 로
# 옮긴다" 는 철회했다 — IPC 는 왕복 2725 ms 중 약 206 ms(7.6 %)라 옮겨도 예산
# 533 ms 를 못 맞추고, 사용자가 원하는 분담만 깨진다.
set -euo pipefail

workspace=/mnt/dev/work/pi05_TO_hybrid

# openpi 는 vendored 서브모듈이고 cwd 로 찾는다 (`serve_rby1_16d.sh` 와 같다).
cd "$workspace/openpi"

# `benchmark.trajopt.serve_safe` 를 import 하려면 작업 루트가 경로에 있어야 한다.
export PYTHONPATH=/mnt/dev/work

export OPENPI_DATA_HOME="$workspace/cache/openpi"
# 이것이 없으면 CUDA graph 캡처가 실패한다.
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
# 이것이 없으면 JAX 가 GPU 105 GB 를 선점해 cuRobo/warp 가 설 자리를 잃는다.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
# 서버는 시뮬레이션을 돌리지 않지만 MJCF 로 로봇 모델을 만들 때 mujoco 를 쓴다.
export MUJOCO_GL=osmesa
# cwd 가 `openpi` 라 `rby1_description/` 를 상대경로로 못 찾는다. **심볼릭 링크를
# 만들지 않는다** — 컨테이너 재시작에 날아가 재생이 전부 죽은 적이 있다.
# `asset_path.py` 가 이 변수를 먼저 본다 (CLAUDE.md "자산 경로").
export AG3S_ASSET_ROOT="$workspace"

ckpt="$workspace/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999"
xml="$workspace/rby1_description/models/rby1a/mujoco/model_transport.xml"

# `--esdf-backend curobo` 는 **반드시** 준다. 기본값이 `legacy`(numpy EsdfBuilder) 라
# 빼먹으면 서버가 조용히 legacy 로 떠서 통합 테스트가 다른 것을 잰다
# (`serve_safe.py:80-82` 가 그 경고를 찍는다 — 로그에서 확인할 것).
exec /mnt/dev/work/.venv-openpi-live/bin/python -u -m benchmark.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora \
  --checkpoint "$ckpt" \
  --model-xml "$xml" \
  --port "${PORT:-8000}" \
  --esdf-backend curobo \
  --voxel 0.020 \
  --fine-voxel 0.005 \
  --tsdf-voxel 0.005 \
  --links arms \
  "$@"
