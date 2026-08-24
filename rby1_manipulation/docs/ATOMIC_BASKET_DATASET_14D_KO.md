# RBY1 원자 과일-바구니 14-D 데이터셋

## 목적

이 파이프라인은 π0.5가 한 prompt에서 하나의 명확한 작업만 수행하도록 학습한다.
하나의 episode는 과일 하나를 바구니에 넣거나 바구니를 한 번 들어 올리는 작업만
포함한다. 내부 MuJoCo body 이름은 기존 호환성을 위해 `crate`를 유지하지만, 모든
학습·추론 prompt와 metadata의 용기 명칭은 `basket`으로 통일한다.

기존 `transport_pack_lift.py`와 `rby1_transport_14d`는 변경하지 않고 별도로
보존한다. 새 scenario는 `rby1_manipulation.tasks.transport_atomic`, collector는
`rby1_manipulation.data.collect_atomic_transport_dataset`이다.

## 기본 2,000 episode 구성

| 계층 | 전체 | train | validation | test |
|---|---:|---:|---:|---:|
| `clean_single` | 1,200 | 960 | 120 | 120 |
| `linked_atomic` | 320 | 256 | 32 | 32 |
| `preloaded_single` | 240 | 192 | 24 | 24 |
| `recovery` | 160 | 128 | 16 | 16 |
| `lift_basket` | 80 | 64 | 8 | 8 |
| 합계 | 2,000 | 1,600 | 200 | 200 |

episode index는 train `0:1600`, validation `1600:1800`, test `1800:2000`의
연속 범위로 저장한다. 수집 완료 시 `meta/info.json`과
`meta/atomic_splits.json`에 두 범위를 기록하며, 같은 sequence group은 split을
넘지 않는다.

과일 투입 1,920개는 apple, banana, orange, pear가 각각 480개다. 각 target은
4개 table slot에 120회씩, 왼팔과 오른팔에 240회씩 배정된다. 16개 fruit layout은
각 124~126회 사용한다.

`linked_atomic`은 100개 2단계 그룹과 40개 3단계 그룹으로 구성한다. 그룹의 각
step은 별도 episode다. 다음 step은 앞 step의 target을 `preloaded_fruits`로
재구성하며 동일한 layout, slot order, scene seed, split을 사용한다. 따라서 긴
trajectory 하나에 여러 prompt를 섞지 않는다.

`preloaded_single`은 target별로 사전 적재 수 1·2·3개를 각각 20개 사용한다.
target은 절대 미리 들어 있지 않다. `lift_basket`은 사전 적재 수 0~4개를 각각
16개 사용하며 각 과일은 40회씩 바구니 안에 등장한다.

## Episode 단계

과일 투입:

```text
initial_hold (>= 1 s)
→ approach
→ pregrasp_align
→ gripper_close
→ grasp_verify
→ lift_from_table
→ transport
→ lower_into_basket
→ release
→ release_verify
→ retreat
→ target_settle
→ return_to_ready
→ terminal_hold (2 s)
→ end
```

release 후 기존 18cm 수직 retreat를 수행한다. target이 내부에서 안정되고 hand가
rim 위 안전거리로 빠진 뒤 사용한 팔을 초기 task-ready 관절 자세로 복귀시킨다.
실측 관절 오차가 0.02rad 이내로 수렴한 뒤 성공 시점을 기록하고, 같은 prompt로
2초간 action을 hold한다. 초기/terminal hold는 `speed_scale`로 줄이지 않는다.

recovery episode는 `empty_close`, `early_close`, `occlusion_reobserve`,
`unsafe_path_replan` 네 종류를 각 40개 사용한다. 충돌을 expert action으로 만들지
않고, close 실패 또는 위험 접근을 중단한 뒤 reopen, retreat, reapproach하여 최종
성공하는 trajectory를 기록한다.

## Frame 및 episode metadata

기존 14-D state/action과 세 policy camera는 유지한다. 현재 14-D는 팔마다 관절
0..5와 gripper를 저장하며, 7번째 관절(`arm_6`)은 scripted IK가 제어한다. 이를
고정하면 basket carry IK 잔차가 약 18~19cm까지 증가하므로 고정하지 않으며, 대신
episode metadata에 `unobserved_arm_6_max_motion_rad`를 기록한다. 실물 adapter에서
이 관절을 어떻게 생성할지는 별도 interface 결정이 필요하다. 카메라는 4:3으로
렌더한 뒤 224x224로 resize하므로 `view_cameras.py` 기본 화면과 동일한 시야를
사용한다. atomic writer에서만 Parquet에
다음 필드를 추가한다.

transport office의 큰 scene extent 때문에 MuJoCo 기본 near plane은 약 22.6cm가
되어 grasp 순간의 gripper와 table을 잘랐다. 공통 모델은 `znear=0.001`을 사용하여
실제 near plane을 약 2.3cm로 낮춘다. 자동 검증은 near plane 3cm 이하, wrist lens의
table clearance 4cm 이상, grasp 구간 중심 광선이 table 영역을 향하는지를 확인한다.

카메라 확인 명령은 기본적으로 transport 모델과 실제 policy 전처리를 사용한다.

```bash
python src/rby1_bringup/view_cameras.py

# 원본 640x480 시야가 필요할 때
python src/rby1_bringup/view_cameras.py --native-view

# 예전 block scene 비교
python src/rby1_bringup/view_cameras.py --model blocks
```

- `phase_index`: frame별 phase ID
- `prompt_timestamp`: prompt가 적용된 시각, 현재 항상 0.0

`meta/atomic_schema.json`은 phase ID와 실패 원인 enum을 정의한다.
`meta/atomic_episodes.jsonl`에는 다음을 기록한다.

- canonical/실제 prompt와 paraphrase 여부
- target/non-target/preloaded fruits
- layout, slot order, 초기 basket/fruit pose
- 사용 팔, grasp attempts/retries, recovery type
- sequence group/step/length와 train/validation/test split
- grasp/release/retreat/return-to-ready/success/terminal hold frame
- terminal hold 길이와 episode 길이
- 성공 여부, 실패 원인과 모든 자동 판정 결과

## 자동 성공 판정

과일 투입은 다음을 모두 만족해야 저장한다.

- target이 초기에는 밖에 있고 최종에는 새로 내부에 있음
- target과 gripper가 완전히 분리됨
- target이 안정되고 terminal hold 마지막 0.5초의 이동량이 10mm 이하
- table 위 non-target의 episode 전체 최대 변위가 20mm 이하
- non-target이 한 번도 새로 바구니에 들어가지 않음
- preloaded fruit가 한 번도 바구니 밖으로 나오지 않음
- gripper가 잘못된 과일을 grasp하지 않음
- hand-target 거리 100mm 이상, hand가 rim보다 100mm 이상 위로 retreat
- 사용한 팔이 초기 task-ready 자세로 복귀하고 최대 관절 오차가 0.02rad 이하
- pregrasp/grasp 중 wrist camera 중심 광선이 table 위를 향하고 lens clearance가 40mm 이상
- table/inter-arm collision 없음
- base drift 0.01m/rad 미만
- terminal hold가 최소 1초 이상 기록됨

실패 rollout은 기본 training dataset에 저장하지 않는다. 성공적인 recovery만
`recovery` 계층에 포함한다.

## 단일 episode 확인

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws

# 빈 바구니에 apple 하나 넣기
src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.tasks.transport_atomic \
  --task place_one \
  --target-fruit apple \
  --task-prompt "put the apple in the basket" \
  --layout-index 0 \
  --slot-order apple banana orange pear \
  --headless

# 다른 세 과일이 이미 든 바구니에 apple만 추가
src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.tasks.transport_atomic \
  --task place_one \
  --target-fruit apple \
  --preloaded banana orange pear \
  --task-prompt "put the apple in the basket" \
  --random-scene --headless

# 의도된 empty-close 후 재접근
src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.tasks.transport_atomic \
  --task place_one \
  --target-fruit pear \
  --scenario-family recovery \
  --recovery-type empty_close \
  --task-prompt "put the pear in the basket" \
  --headless

# 바구니 들기
src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.tasks.transport_atomic \
  --task lift_basket \
  --preloaded apple banana \
  --task-prompt "lift the basket" \
  --headless
```

## Schedule 확인과 수집

```bash
# 파일을 저장하지 않고 2,000개 분포와 command 확인
src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.data.collect_atomic_transport_dataset \
  --output-dir /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_atomic_basket_14d_v2 \
  --dry-run --dry-run-limit 12

# 실제 수집
PYTHONUNBUFFERED=1 src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.data.collect_atomic_transport_dataset \
  --output-dir /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_atomic_basket_14d_v2 \
  --speed-scale 1.25 \
  --timeout 300 \
  --max-attempts 5
```

수집기는 성공 episode만 저장하며 중단 후 같은 명령으로 재개한다. 초기
randomization은 fruit grid ±6mm, basket xy ±10mm, basket yaw ±3°, basket mass
0.65~1.0kg, friction multiplier 0.9~1.2다.

## Paraphrase 확장

기본 2,000개는 canonical prompt만 사용한다. canonical 평가가 충분한 뒤
`--include-paraphrases`를 사용하면 240개를 추가한 2,240개 계획을 만든다.

```text
place the {fruit} inside the basket
move the {fruit} into the basket
pick up the {fruit} and put it in the basket
```

각 과일×표현 조합은 20개다. 기본 계획과 다른 output directory를 사용해야 한다.

## 데이터 품질 검증

```bash
src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.data.validate_atomic_dataset \
  --dataset /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_atomic_basket_14d_v2

# MP4 실제 frame 수도 전부 디코딩해 검사
src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.data.validate_atomic_dataset \
  --dataset /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_atomic_basket_14d_v2 \
  --check-video-frames
```

validator는 episode 수, 14-D schema, Parquet/MP4 존재, timestamp, initial/terminal
hold, event 순서, semantic check, 과일·계층·split 분포를 검사한다.

## 추론 평가

평가 rollout을 `atomic_episodes.jsonl`과 같은 결과 schema로 기록한 뒤 다음 명령을
사용한다.

```bash
src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.evaluation.atomic_transport \
  --results evaluation/atomic_rollouts.jsonl \
  --output evaluation/atomic_metrics.json
```

다음을 계산한다.

- prompt compliance rate
- target placement rate
- unnecessary fruit insertion rate
- wrong-target grasp/non-target displacement rate
- complete release/safe retreat/terminal hold rate
- 성공 후 추가 gripper close 비율
- sequence group 전체 완료율
- prompt별 성공률과 실패 원인 분포

## 1,989 episode 확정본과 fine-tuning

2026-08-25 수집본은 2,000개 계획 중 성공한 1,989개만 사용한다. 실패 rollout은
저장되지 않았고, 누락 11개를 채우기 위해 데이터 수집을 더 수행하지 않는다.

```text
전체       1,989 episodes / 622,261 frames
train      1,591 episodes / 497,754 frames / index 0:1591
validation   199 episodes /  62,259 frames / index 1591:1790
test         199 episodes /  62,248 frames / index 1790:1989
```

`finalize_atomic_dataset`은 실제 저장 순서를 검사한 뒤 `meta/info.json`과
`meta/atomic_splits.json`을 원자적으로 기록한다. intentionally partial 상태와 누락
plan index도 보존하며, parquet 옆에 `.bak`을 만들지 않는다.

```bash
src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.data.finalize_atomic_dataset \
  --dataset datasets/rby1_atomic_basket_14d_v2

src/rby1_manipulation/.venv/bin/python \
  -m rby1_manipulation.data.validate_atomic_dataset \
  --dataset datasets/rby1_atomic_basket_14d_v2 \
  --check-video-frames
```

이번 writer는 수집 시점부터 parquet timestamp를 `arange(n) / 15`로 기록한다.
따라서 이전 데이터셋에 사용한 `patch_dataset_timestamps.py`를 다시 실행하지 않는다.
전송 전후 `preflight_atomic_finetune.py`가 모든 parquet의 timestamp, 14-D shape,
episode index, 파일 수, split, `.bak`/`.tmp` 부재를 재검사한다.

OpenPI 설정 `pi05_rby1_atomic_lora`는 repo
`local/rby1_atomic_basket_14d_v2`와 train episode `0..1590`만 로드한다. 따라서
normalization과 optimization에서 기대하는 HuggingFace train split은 전체
622,261 frame이 아니라 정확히 497,754 frame이다. validation/test를 학습에
포함하지 않는다. 새 repo id, config name, HuggingFace cache 경로와 checkpoint
experiment name을 사용하여 이전 `rby1_dataset_v1` 캐시·norm·checkpoint와 격리한다.
