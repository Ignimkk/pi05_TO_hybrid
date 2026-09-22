# RB-Y1 randomized 16-D pick-and-place 수집

기존 `rby1_14` atomic collector와 별개로 `rby1_randomized_pick_place_16d_v1`을 수집한다. 각 episode는 reset된 target/basket pose에서 waypoint와 IK를 다시 계산하며 성공한 rollout만 저장한다.

## 데이터 정의

- 관측: `cam_high`, `cam_left_wrist`, `cam_right_wrist` RGB(224×224), 16-D robot state, language task
- state/action 순서: `[left_arm_0..6, left_gripper, right_arm_0..6, right_gripper]`
- state: 측정 qpos, action: absolute joint-position control, gripper: 0–1
- FPS: 15
- episode 수: apple/banana/orange/pear 각 500, 총 2,000
- split: train 1,600 / validation 200 / test 200
- OpenPI 학습 config: `pi05_rby1_randomized_pick_place_16d_lora` (train episode `0..1599`)

기본 randomization config는 `src/rby1_manipulation/config/randomized_pick_place.json`이다. 각 toggle은 독립적이며, 꺼진 항목은 nominal 값을 정확히 사용한다. Randomized collector의 준비자세는 기존 teleop keyframe보다 양쪽 `arm_1`을 바깥으로 각각 5° 이동한다(손목 Y 간격 약 45.9 cm → 53.4 cm). 그 준비자세를 기준으로 target X ±2 cm/Y ±3 cm, basket XY ±3 cm, target yaw ±45°, 14개 arm joint ±2°를 sampling한다. lighting, texture, camera, friction, mass, damping은 변경하지 않는다.

## 100-reset 사전검사

저장 경로를 새 디렉터리로 지정한다.

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
MUJOCO_GL=osmesa \
PYTHONPATH=src/rby1_manipulation/src \
src/rby1_manipulation/.venv/bin/python -m \
  rby1_manipulation.data.preflight_randomized_pick_place \
  --config src/rby1_manipulation/src/rby1_manipulation/config/randomized_pick_place.json \
  --samples 100 \
  --seed 20260918 \
  --output-dir outputs/randomized_pick_place_preflight_v4_banana_clearance
```

`randomization_summary.json`의 `passed`가 true이고 config fingerprint가 현재 config와 일치해야 collector가 시작된다. 범위를 수정했다면 기존 report를 재사용할 수 없으며 100-reset 검사를 다시 수행해야 한다.

## smoke 수집과 검증

```bash
MUJOCO_GL=osmesa \
PYTHONPATH=src/rby1_manipulation/src \
src/rby1_manipulation/.venv/bin/python -m \
  rby1_manipulation.data.collect_randomized_pick_place_dataset \
  --output-dir datasets/rby1_randomized_pick_place_16d_v1_smoke_v3_wider_ready \
  --randomization-config src/rby1_manipulation/src/rby1_manipulation/config/randomized_pick_place.json \
  --preflight-report outputs/randomized_pick_place_preflight_v4_banana_clearance/randomization_summary.json \
  --smoke-per-fruit 1

PYTHONPATH=src/rby1_manipulation/src \
src/rby1_manipulation/.venv/bin/python -m \
  rby1_manipulation.data.validate_randomized_pick_place_dataset \
  --dataset datasets/rby1_randomized_pick_place_16d_v1_smoke_v3_wider_ready \
  --check-video-frames
```

## 전체 2,000개 수집: tmux 권장

전체 수집처럼 장시간 실행하는 작업은 다음처럼 tmux 세션에서 시작한다.

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
tmux new-session -s rby1_16d_collect
```

tmux 안에서:

```bash
MUJOCO_GL=osmesa \
PYTHONPATH=src/rby1_manipulation/src \
src/rby1_manipulation/.venv/bin/python -m \
  rby1_manipulation.data.collect_randomized_pick_place_dataset \
  --output-dir datasets/rby1_randomized_pick_place_16d_v1 \
  --randomization-config src/rby1_manipulation/src/rby1_manipulation/config/randomized_pick_place.json \
  --preflight-report outputs/randomized_pick_place_preflight_v4_banana_clearance/randomization_summary.json \
  2>&1 | tee outputs/rby1_randomized_pick_place_16d_v1.collect.log
```

`Ctrl-b d`로 detach하고 `tmux attach -t rby1_16d_collect`로 복귀한다. 프로세스가 종료된 경우 같은 명령을 다시 실행하면 성공 episode 다음부터 resume한다. 단, schema version, schedule, FPS/hold 설정 또는 randomization fingerprint가 달라지면 안전하게 중단한다. 다른 설정을 쓸 때는 반드시 새 dataset 디렉터리를 사용한다.

수집 완료 후:

```bash
PYTHONPATH=src/rby1_manipulation/src \
src/rby1_manipulation/.venv/bin/python -m \
  rby1_manipulation.data.validate_randomized_pick_place_dataset \
  --dataset datasets/rby1_randomized_pick_place_16d_v1 \
  --check-video-frames
```

실패 시도는 `randomized_collection_attempts.jsonl`, 성공 mapping은 `randomized_collection_manifest.jsonl`, resume 상태는 `randomized_collection_stats.json`에 기록된다.

## 완료된 전체 데이터셋

2026-09-22 기준 `datasets/rby1_randomized_pick_place_16d_v1`에 2,000개 episode 수집 및 재인덱싱을 완료했다. train/validation/test는 각각 `0:1600`, `1600:1800`, `1800:2000`이며, 1,000 episode 단위로 `chunk-000`과 `chunk-001`에 저장된다. 총 frame 수는 621,892이고 세 camera의 MP4 6,000개를 모두 frame 단위로 검증했다.

초기 수집에서 실패한 banana episode 36개는 finger/table clearance를 확보하도록 banana grasp 높이 하한을 22 mm에서 30 mm로 수정한 뒤 다시 계산한 trajectory로 보충했다. 해당 episode와 planner revision은 `meta/randomized_schema.json` 및 `reindex_report.json`에 기록된다. 수정 후 사전검사는 `outputs/randomized_pick_place_preflight_v4_banana_clearance`에서 100/100 유효 reset으로 통과했다.

재인덱싱 전 append-order 데이터는 복구용으로 `datasets/rby1_randomized_pick_place_16d_v1_pre_reindex_backup_20260922`에 보존되어 있다. 영상은 완료본과 hard link를 공유하므로 백업 삭제 전에는 두 경로가 같은 영상 inode를 참조할 수 있다.
