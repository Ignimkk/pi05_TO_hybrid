# RB-Y1 randomized 16-D pick-and-place 데이터 수집 및 π0.5 fine-tuning runbook

이 문서는 RB-Y1 MuJoCo 환경에서 randomized fruit-to-basket expert demonstration을 설계하고, LeRobot 형식으로 수집·검사한 뒤 GPU 서버로 전송하여 OpenPI π0.5 LoRA fine-tuning을 수행한 전체 과정을 재현하기 위한 운영 기록이다.

문서의 대상 데이터셋은 `rby1_randomized_pick_place_16d_v1`이다. 기존 14-D atomic 데이터셋과 수집기에는 영향을 주지 않는 별도 파이프라인이다.

## 1. 완료 상태 요약

로컬 완료 데이터셋:

```text
/home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_randomized_pick_place_16d_v1
```

| 항목 | 결과 |
|---|---:|
| 성공 episode | 2,000 |
| 전체 수집 시도 | 3,422 |
| 총 frame | 621,892 |
| Parquet | 2,000 |
| MP4 | 6,000 (3 cameras × 2,000 episodes) |
| FPS | 15 |
| 이미지 | 224×224 RGB |
| state/action | 16-D |
| target | apple/banana/orange/pear 각 500 |
| split | train 1,600 / validation 200 / test 200 |
| chunk | 1,000 episodes 단위, `chunk-000`, `chunk-001` |
| 로컬 크기 | 약 3.9 GB |
| randomization fingerprint | `47f91ebc478524c2` |

서버 데이터셋:

```text
/mnt/dev/work/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1
```

OpenPI repo ID용 링크:

```text
/mnt/dev/work/pi05_TO_hybrid/data/local/rby1_randomized_pick_place_16d_v1
  -> ../rby1_randomized_pick_place_16d_v1
```

서버에서 2,000 Parquet, 6,000 MP4, 621,892 frames, 15 FPS 및 split을 다시 확인했다. `data/` 내부의 `*.bak`, `*.tmp`는 0개다.

Normalization은 완료되었으며 결과는 다음 경로에 있다.

```text
/mnt/dev/work/pi05_TO_hybrid/openpi/assets/
  pi05_rby1_randomized_pick_place_16d_lora/
  local/rby1_randomized_pick_place_16d_v1/norm_stats.json
```

Fine-tuning은 다음 실험 이름으로 실행한다. 이 문서 작성 시점에는 XLA 재시작 run이 정상 진행 중이므로 완료 여부는 반드시 서버에서 별도로 확인한다.

```text
rby1_randomized_pick_place_16d_30k_xla_retry_20260923
```

## 2. 전체 파이프라인

```text
시나리오/분포 설계
  → 100-reset randomization preflight
  → 과일별 smoke rollout
  → tmux에서 전체 demonstration 수집
  → 실패 plan 보충 및 plan 순서로 재인덱싱
  → Parquet/MP4/metadata 전체 검증
  → rsync로 GPU 서버 전송 및 checksum dry-run
  → LeRobot/OpenPI 실제 loader 확인
  → train split만으로 norm_stats 계산
  → π0.5 base + LoRA 30K fine-tuning
  → checkpoint, norm_stats, config revision 함께 보관
```

## 3. 주요 코드와 산출물 위치

| 역할 | 경로 |
|---|---|
| randomization 설정 | `src/rby1_manipulation/src/rby1_manipulation/config/randomized_pick_place.json` |
| scene sampling/validity/IK dry-run | `src/rby1_manipulation/src/rby1_manipulation/simulation/randomized_pick_place.py` |
| reset 및 object/robot pose 적용 | `src/rby1_manipulation/src/rby1_manipulation/simulation/fruit_grid.py` |
| waypoint/IK 및 episode 실행 | `src/rby1_manipulation/src/rby1_manipulation/tasks/transport_atomic.py` |
| frame sampling/timestamp 생성 | `src/rby1_manipulation/src/rby1_manipulation/data/recording.py` |
| LeRobot Parquet/MP4 writer | `src/rby1_manipulation/src/rby1_manipulation/data/episode.py` |
| 100-reset preflight | `src/rby1_manipulation/src/rby1_manipulation/data/preflight_randomized_pick_place.py` |
| 전체 collector/schedule/resume | `src/rby1_manipulation/src/rby1_manipulation/data/collect_randomized_pick_place_dataset.py` |
| recovery reindex | `src/rby1_manipulation/src/rby1_manipulation/data/reindex_randomized_pick_place_dataset.py` |
| 최종 validator | `src/rby1_manipulation/src/rby1_manipulation/data/validate_randomized_pick_place_dataset.py` |
| collector 테스트 | `src/rby1_manipulation/tests/test_randomized_pick_place_collection.py` |
| OpenPI 학습 config | `src/openpi/src/openpi/training/config.py` |
| OpenPI ALOHA 16-D 변환 | `src/openpi/src/openpi/policies/aloha_policy.py` |
| OpenPI 16-D 변환 테스트 | `src/openpi/src/openpi/policies/aloha_policy_test.py` |

완료 데이터셋에서 중요한 기록은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `randomized_collection_plan.json` | 고정 schedule, seed, prompt, config fingerprint |
| `randomized_collection_attempts.jsonl` | 성공/실패를 포함한 모든 rollout 시도와 실패 원인 |
| `randomized_collection_manifest.jsonl` | 성공하여 저장된 episode mapping |
| `randomized_collection_stats.json` | plan별 attempt 수, 성공 여부, resume 상태 |
| `meta/info.json` | LeRobot schema, FPS, totals, split |
| `meta/tasks.jsonl` | `task_index`와 실제 language instruction 연결 |
| `meta/randomized_episodes.jsonl` | scene, seed, pose, 초기 state, event frame, success metadata |
| `meta/randomized_schema.json` | 16-D schema, fingerprint, 제외한 randomization, recovery revision |
| `meta/randomized_splits.json` | contiguous split 범위 |
| `reindex_report.json` | recovery 이후 old-to-new episode mapping |

## 4. 시나리오 설계

### 4.1 Task semantics

로봇은 테이블 위 네 과일 중 language instruction이 지시한 과일 하나를 집어 basket에 넣는다. 내부 MuJoCo body 이름은 기존 코드 호환을 위해 `crate`를 유지하지만 사용자 의미와 prompt에서는 `basket`을 사용한다.

Target이 아닌 세 과일은 distractor로 남기며 위치와 orientation을 randomize하지 않는다. Lighting, texture, camera calibration/extrinsics, friction, mass, damping도 이번 데이터셋에서는 변경하지 않는다.

### 4.2 매 episode에서 새 trajectory를 만드는 이유

이 데이터셋은 과거 trajectory를 새로운 scene에서 replay하지 않는다. 매 episode마다 다음 순서를 수행한다.

```text
reset
→ target/basket/robot initial state sampling
→ settle 및 scene validity 검사
→ 현재 target body pose에서 grasp frame 재계산
→ 현재 target/basket 위치로 waypoint 생성
→ IK trajectory 계산 및 실행
→ rollout success 검사
→ 성공 episode만 저장
```

Target yaw는 현재 물체의 world-Z orientation에 합성된다. 따라서 banana처럼 방향성이 큰 물체도 현재 yaw에 맞춘 grasp frame을 사용한다. Pick waypoint는 실행 직전에 현재 object pose로 만들고, place waypoint는 grasp 성공 후 현재 basket pose로 만든다.

실행 phase는 코드상 다음처럼 세분화되어 있다.

```text
Initial hold
→ Approach / Pre-grasp align
→ Gripper close / Grasp verify
→ Lift from table
→ Transport / Lower into basket
→ Release / Release verify
→ Retreat / Target settle
→ Return to ready
→ Terminal hold
```

### 4.3 Randomization 설정

현재 확정 설정은 다음과 같다.

| 항목 | 현재 값 | 독립 toggle |
|---|---:|---|
| target position | nominal 기준 X ±2 cm, Y ±3 cm | `target_position.enabled` |
| basket position | nominal 기준 X/Y ±3 cm | `goal_position.enabled` |
| target yaw | ±45° | `target_orientation.enabled` |
| robot initial joints | 양팔 14 arm joints 각각 ±2° | `robot_initial_configuration.enabled` |
| language | target별 3 paraphrases | `language_instruction.enabled` |
| scene sampling 한도 | 100 candidates/reset | 해당 없음 |
| settle | 1.5 s | 해당 없음 |
| object clearance | 7.5 cm | 해당 없음 |
| accepted IK residual | 최대 15 mm | 해당 없음 |

Robot ready pose는 MuJoCo `teleop` keyframe을 기준으로 양팔을 더 벌린 자세다.

```text
left arm_1  : +5°
right arm_1 : -5°
other joints: 0° offset
```

이 변경으로 wrist 간격이 약 45.9 cm에서 53.4 cm로 넓어졌다. Joint noise는 이 ready pose 위에 더한다. Gripper는 open 상태로 고정한다.

Language templates:

```text
Put the {target} in the basket.
Place the {target} into the basket.
Pick up the {target} and place it in the basket.
```

Toggle을 끄면 해당 값은 nominal과 정확히 같아야 한다. Ablation 시에는 config를 수정한 뒤 반드시 새로운 preflight report와 새로운 dataset 디렉터리를 사용한다.

### 4.4 Scene validity와 rollout success

Candidate scene은 다음을 검사한다.

- Target과 basket footprint가 허용 table/workspace 범위 안에 있는가
- Target과 basket, target과 distractor 사이 clearance가 충분한가
- 초기 14 arm joints가 joint limit 안에 있는가
- 초기 arm/self/inter-arm/environment 또는 prop 간 잘못된 collision이 없는가
- 1.5초 settle 후 target과 basket 속도가 허용값 이하인가
- Pre-grasp, grasp, lift, pre-place, place, retreat의 dry-run IK residual이 모두 15 mm 이하인가

Rollout은 추가로 다음을 검사한다.

- 올바른 target을 실제로 grasp했는가
- finger/table 및 inter-arm collision이 없는가
- wrist camera view가 유효한가
- target이 basket 안에 새로 들어갔고 완전히 release되었는가
- target이 안정적으로 settle했는가
- distractor가 움직이거나 basket에 들어가지 않았는가
- retreat clearance와 return-to-ready가 안전한가
- terminal hold 동안 결과가 유지되는가

`SUCCESS=False` rollout은 Parquet/MP4에 저장하지 않는다. 실패 원인과 sampled scene만 attempt manifest에 남고 다음 retry는 새로운 seed로 scene과 IK를 다시 계산한다.

## 5. Schedule과 LeRobot schema

### 5.1 균형 schedule

Seed `20260918`로 다음 균형을 고정한다.

- 과일별 500 episodes
- 과일별 train/validation/test = 400/50/50
- 전체 split = `0:1600`, `1600:1800`, `1800:2000`
- 16 layouts 각각 125회
- 과일별 target slot 0/1/2/3 각각 125회
- 사용 arm left/right 각각 1,000회
- 각 과일·split 안에서 세 paraphrase 수 차이는 최대 1

Collector 완료 후 append 순서가 아니라 `plan_index` 순서로 정렬되어야 split이 올바르게 유지된다.

### 5.2 Timestep data

각 Parquet row에는 최소 다음 필드가 있다.

```text
observation.state
action
timestamp
frame_index
episode_index
index
task_index
next.done
next.reward
phase_index
prompt_timestamp
```

세 RGB stream은 MP4로 저장한다.

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
```

State/action 순서는 동일하다.

```text
[left_arm_0..6, left_gripper, right_arm_0..6, right_gripper]
```

- State: 측정 qpos
- Action: absolute joint-position control
- Gripper: 0–1 normalized absolute value
- Language: Parquet의 `task_index`가 `meta/tasks.jsonl` 문장으로 연결됨

Episode metadata에는 requested/settled target 및 basket pose, 초기 16-D state, raw 14 arm joint, seed, randomization 값/toggle, sampling attempts, validity, phase event frame, used arm, prompt/canonical prompt, success가 기록된다.

## 6. 로컬 사전검사

작업 루트:

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
```

### 6.1 관련 테스트

```bash
PYTHONPATH=src/rby1_manipulation/src \
src/rby1_manipulation/.venv/bin/python -m pytest -q \
  src/rby1_manipulation/tests/test_randomized_pick_place_collection.py

cd src/openpi
.venv/bin/python -m pytest -q \
  src/openpi/policies/aloha_policy_test.py
cd /home/mk/dev_ws/vla/pi0_TO_ws
```

OpenPI 테스트는 7 joints/arm delta mask, gripper absolute 유지, 16→32-D padding, 16-D output slice 및 기존 6 joints/arm 기본값 보존을 확인한다.

### 6.2 100-reset preflight

최종 banana clearance 변경 후 사용한 검사:

```bash
MUJOCO_GL=osmesa \
PYTHONPATH=src/rby1_manipulation/src \
src/rby1_manipulation/.venv/bin/python -m \
  rby1_manipulation.data.preflight_randomized_pick_place \
  --config src/rby1_manipulation/src/rby1_manipulation/config/randomized_pick_place.json \
  --samples 100 \
  --seed 20260918 \
  --output-dir outputs/randomized_pick_place_preflight_v4_banana_clearance
```

산출물:

```text
randomization_samples.jsonl
randomization_summary.json
randomization_xy.png
```

최종 결과:

```text
passed                         true
valid resets                   100/100
candidate draws                261
candidate acceptance rate      38.31%
sampling attempts p95          7.05
sampling attempts max          14
accepted max IK residual       14.10 mm
config fingerprint             47f91ebc478524c2
```

현재 구현의 `passed` 조건은 100/100 reset 성공, candidate acceptance ≥20%, attempts p95 ≤20, **accepted scene**의 IK residual ≤15 mm, 활성 범위의 p05–p95 span ≥60%, 모든 language template 관측이다.

주의: `passed=true`는 candidate rejection이 0이라는 뜻이 아니다. 최종 검사에서도 workspace, settle, overlap, 초기 prop collision, IK unreachable 후보가 reject된 뒤 다시 sampling되었다. 중요한 것은 reject candidate가 dataset에 들어가지 않고 100회 모두 sampling 한도 안에서 유효 scene을 얻었다는 점이다. Candidate-level IK/collision reject까지 0을 요구하려면 validator 기준을 별도로 강화해야 한다.

Config 값이나 ready pose를 한 글자라도 바꾸면 fingerprint가 바뀐다. Collector는 fingerprint가 다른 preflight report를 거부한다.

### 6.3 Smoke dataset

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

Smoke test에서는 네 과일이 각각 최소 한 번 성공하고, seed가 다른 episode의 requested pose 및 action trajectory가 실제로 달라지는지 확인한다.

## 7. 전체 데이터 수집

### 7.1 tmux에서 시작

전체 수집은 약 하루 이상 걸릴 수 있으므로 shell 연결에 직접 매달아 실행하지 않는다.

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
tmux new-session -s rby1_16d_collect
```

tmux 내부:

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

```text
detach:  Ctrl-b d
attach:  tmux attach -t rby1_16d_collect
목록:    tmux ls
```

실제 최초 전체 collection plan에는 동일 fingerprint의 `v3_wider_ready` preflight 경로가 기록되어 있다. Banana clearance 수정 후 recovery와 최종 확인은 `v4_banana_clearance`를 사용했다. 설정 fingerprint는 둘 다 `47f91ebc478524c2`이며, banana grasp 높이 변경은 randomization config가 아니라 planner revision이다.

### 7.2 Resume 규칙

수집이 중단되면 먼저 다음을 확인한다.

```bash
tmux ls
pgrep -af collect_randomized_pick_place_dataset
tail -100 outputs/rby1_randomized_pick_place_16d_v1.collect.log
jq '[.episodes[] | select(.success == true)] | length' \
  datasets/rby1_randomized_pick_place_16d_v1/randomized_collection_stats.json
```

프로세스가 실제로 종료된 경우 같은 명령을 다시 실행한다. Collector는 성공 plan을 건너뛰고 미완료 plan부터 재개한다.

다음 값이 기존 plan과 다르면 resume하지 않고 중단한다.

- schema/version/mode
- schedule seed와 2,000 episode schedule
- randomization config fingerprint
- FPS, speed scale, initial/terminal hold
- smoke/full 구분

다른 설정의 실험은 반드시 새 output directory를 사용한다. 실행 중인 collector 위에 두 번째 collector를 띄우면 episode index와 metadata가 손상될 수 있다.

### 7.3 실제 recovery 이력

최초 run은 최대 10 attempts에서 1,964/2,000 성공으로 종료되었다. 실패한 36개는 모두 banana 관련 plan이었다. Finger/table clearance를 확보하기 위해 banana grasp 높이 하한을 22 mm에서 30 mm로 수정하고, 각 plan을 새로운 seed와 새 trajectory로 보충했다.

Recovery 후 결과:

```text
successful plans  2,000/2,000
total attempts     3,422
max attempts/plan  12
```

Recovery episode가 append되었기 때문에 원본 episode index는 plan 순서와 달랐다. 다음 reindex 도구로 새 디렉터리를 만들었다.

```bash
PYTHONPATH=src/rby1_manipulation/src \
src/rby1_manipulation/.venv/bin/python -m \
  rby1_manipulation.data.reindex_randomized_pick_place_dataset \
  --source datasets/rby1_randomized_pick_place_16d_v1 \
  --output datasets/rby1_randomized_pick_place_16d_v1_reindexed
```

검증 후 reindexed 결과를 공식 `rby1_randomized_pick_place_16d_v1`으로 사용했고, 이전 append-order 데이터는 다음 경로에 보존했다.

```text
datasets/rby1_randomized_pick_place_16d_v1_pre_reindex_backup_20260922
```

Reindexer는 MP4를 가능하면 hard link한다. 따라서 백업과 완료본의 같은 영상이 동일 inode를 공유할 수 있다. 한쪽 MP4를 in-place 수정하면 다른 쪽도 바뀔 수 있으므로 백업을 독립 사본으로 생각하면 안 된다.

## 8. 최종 데이터 검사

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
PYTHONPATH=src/rby1_manipulation/src \
src/rby1_manipulation/.venv/bin/python -m \
  rby1_manipulation.data.validate_randomized_pick_place_dataset \
  --dataset datasets/rby1_randomized_pick_place_16d_v1 \
  --check-video-frames \
  2>&1 | tee outputs/rby1_randomized_pick_place_16d_v1.video_validation.log
```

Validator는 다음을 확인한다.

- plan/info/episode/randomized metadata 개수 일치
- `rby1_16` schema와 정확한 16-D field 순서
- 224×224 RGB, 15 FPS, 세 camera 존재
- 2,000 Parquet 및 6,000 MP4 존재
- 모든 Parquet timestamp가 `frame_index / 15` grid에 있음
- Parquet row 수, episode length, MP4 decoded frame 수 일치
- `task_index`가 실제 prompt로 올바르게 resolve됨
- 성공하고 유효한 scene만 저장됨
- 서로 다른 seed에서 scene pose와 IK action이 달라짐
- 1,600/200/200 contiguous split 및 fruit/arm 균형
- dataset이 `plan_index` 순서로 재인덱싱됨

완료 로그의 최종 메시지:

```text
video frames checked: 6000/6000
randomized dataset validation OK
```

추가로 dataset 내부에 backup/temp 파일이 없는지 확인한다.

```bash
find datasets/rby1_randomized_pick_place_16d_v1 \
  -type f \( -name '*.bak' -o -name '*.tmp' \) -print
```

출력이 없어야 한다.

## 9. GPU 서버로 전송

현재 접속 경로:

```text
bastion:   blunex@ai.amrc.kr:21151
container: root@172.21.121.112:30275
```

### 9.1 전송 전 서버 코드 주의사항

서버에는 부모 저장소와 `openpi/` 내부 저장소가 별도로 있다. 각각 상태를 확인한다.

```bash
ssh -J blunex@ai.amrc.kr:21151 -p 30275 root@172.21.121.112
cd /mnt/dev/work/pi05_TO_hybrid
git status
git -C openpi status
```

확인 당시 부모 저장소의 `rby1_bringup/pi05_infer.py`는 사용자 수정 상태였다. 이 파일을 reset/restore하면 안 된다. OpenPI 저장소만 clean이고 fast-forward 가능할 때 `git pull --ff-only`를 사용한다. 재현 당시 서버 revision은 다음과 같다.

```text
parent repository: b154f3dea8a888e3ef7c12887223c3088313725d
openpi repository: 94382c8ea21d5b738f2d5e7db254b4669c19706d
```

### 9.2 목적지 생성과 rsync

```bash
ssh -J blunex@ai.amrc.kr:21151 -p 30275 root@172.21.121.112 \
  'mkdir -p /mnt/dev/work/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1'

rsync -aH --partial --info=progress2 \
  --exclude='*.bak' --exclude='*.tmp' \
  -e "ssh -J blunex@ai.amrc.kr:21151 -p 30275" \
  /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_randomized_pick_place_16d_v1/ \
  root@172.21.121.112:/mnt/dev/work/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/
```

MP4는 이미 압축되어 있으므로 `-z`는 사용하지 않는다. `--partial` 덕분에 중단되면 같은 명령으로 이어서 전송할 수 있다. Source 경로 끝의 `/`는 디렉터리 내용만 목적지에 복사한다는 의미다.

### 9.3 내용 checksum dry-run

```bash
rsync -anrc --delete --itemize-changes \
  --exclude='*.bak' --exclude='*.tmp' \
  -e "ssh -J blunex@ai.amrc.kr:21151 -p 30275" \
  /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_randomized_pick_place_16d_v1/ \
  root@172.21.121.112:/mnt/dev/work/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/
```

출력이 없으면 파일 내용이 일치한다. 반드시 `-n` dry-run 상태에서 `--delete`를 사용한다. 서버에서 dataset 자체를 수정한 뒤에는 로컬 source로 다시 덮어쓰지 않는다.

### 9.4 서버 측 기본 확인

```bash
cd /mnt/dev/work/pi05_TO_hybrid

jq '{total_episodes,total_frames,fps,splits}' \
  data/rby1_randomized_pick_place_16d_v1/meta/info.json

find data/rby1_randomized_pick_place_16d_v1/data -name '*.parquet' | wc -l
find data/rby1_randomized_pick_place_16d_v1/videos -name '*.mp4' | wc -l
find data/rby1_randomized_pick_place_16d_v1/data \
  -type f \( -name '*.bak' -o -name '*.tmp' \) | wc -l
```

기대값은 `2000`, `621892`, `15`, `2000`, `6000`, backup/temp `0`이다.

## 10. LeRobot/OpenPI 데이터 연결

OpenPI config의 repo ID는 다음과 같다.

```text
local/rby1_randomized_pick_place_16d_v1
```

`HF_LEROBOT_HOME=/mnt/dev/work/pi05_TO_hybrid/data`일 때 이 ID가 실제 데이터셋을 가리키도록 링크를 만든다.

```bash
cd /mnt/dev/work/pi05_TO_hybrid
mkdir -p data/local
ln -s ../rby1_randomized_pick_place_16d_v1 \
  data/local/rby1_randomized_pick_place_16d_v1

readlink -f data/local/rby1_randomized_pick_place_16d_v1
```

기대 결과:

```text
/mnt/dev/work/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1
```

이미 링크가 있으면 삭제 후 재생성하지 말고 먼저 `readlink -f`로 대상을 확인한다.

OpenPI loader가 생성하는 학습 batch의 기대 shape는 다음과 같다.

```text
three RGB cameras : (32, 224, 224, 3)
state             : (32, 32)
actions           : (32, 50, 32)
```

원본은 16-D지만 π0.5 내부 model dimension에 맞춰 zero-padding되어 32-D가 된다. 이 loader 초기화가 성공하면 LeRobot의 timestamp 기반 video lookup까지 실제로 수행된 것이다.

## 11. Normalization

### 11.1 OpenPI data transform

학습 config `pi05_rby1_randomized_pick_place_16d_lora`는 다음을 수행한다.

- `episodes=range(1600)`: train split만 사용
- `arm_joint_dim=7`: 양팔 7 joints + gripper
- `adapt_to_pi=False`: ALOHA hardware-specific joint 변환 미사용
- Delta mask: `[7 joint delta, gripper absolute] × 2`
- 세 camera를 `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`로 repack
- 16-D state/action을 π0.5 내부 32-D로 padding
- model output은 16-D로 slice한 뒤 joint delta를 absolute action으로 복원
- Quantile normalization 사용

Validation/test 400 episodes는 norm stats와 optimizer 양쪽에서 제외된다.

### 11.2 실행

장시간 작업이므로 tmux를 사용한다.

```bash
cd /mnt/dev/work/pi05_TO_hybrid
mkdir -p logs cache/openpi

tmux new-session -d -s rby1_16d_normstats \
  "cd /mnt/dev/work/pi05_TO_hybrid/openpi && \
   export HF_LEROBOT_HOME=/mnt/dev/work/pi05_TO_hybrid/data && \
   export OPENPI_DATA_HOME=/mnt/dev/work/pi05_TO_hybrid/cache/openpi && \
   export PYTHONUNBUFFERED=1 && \
   exec .venv/bin/python scripts/compute_norm_stats.py \
     --config-name pi05_rby1_randomized_pick_place_16d_lora \
   2>&1 | tee /mnt/dev/work/pi05_TO_hybrid/logs/rby1_16d_normstats_fast.log"
```

진행 확인:

```bash
tmux attach -t rby1_16d_normstats
tail -f /mnt/dev/work/pi05_TO_hybrid/logs/rby1_16d_normstats_fast.log
```

정상 완료 기준:

```text
train episodes                         1,600
train frames                           497,498
batch size                             32
full normalization batches            15,546
log 마지막                            Writing stats to: ...
norm_stats state mean length           16
norm_stats actions mean length         16
```

통계 파일 검증:

```bash
cd /mnt/dev/work/pi05_TO_hybrid/openpi
.venv/bin/python -c '
import json
from pathlib import Path
p = Path("assets/pi05_rby1_randomized_pick_place_16d_lora/local/rby1_randomized_pick_place_16d_v1/norm_stats.json")
s = json.loads(p.read_text())["norm_stats"]
assert len(s["state"]["mean"]) == 16
assert len(s["actions"]["mean"]) == 16
print("norm stats verified: 16-D state/action")
'
```

`norm_stats.json`은 해당 checkpoint와 항상 함께 보관한다. 다른 dataset revision의 stats를 같은 repo ID 아래 덮어쓰면 안 된다.

## 12. π0.5 LoRA fine-tuning

### 12.1 설정

| 항목 | 값 |
|---|---|
| Config | `pi05_rby1_randomized_pick_place_16d_lora` |
| Base checkpoint | `gs://openpi-assets/checkpoints/pi05_base/params` |
| Vision/language | `gemma_2b_lora` |
| Action expert | `gemma_300m_lora` |
| Train episodes | `0..1599` |
| Batch size | 32 |
| Steps | 30,000 |
| Warmup | 1,000 |
| Peak LR | `5e-5` |
| Final LR | `5e-6` |
| Schedule | cosine decay |
| Save interval | 5,000 |
| EMA | disabled |
| WandB | disabled |

### 12.2 실행

Fine-tuning도 tmux에서 실행한다.

```bash
cd /mnt/dev/work/pi05_TO_hybrid
tmux new-session -s rby1_16d_train
```

tmux 내부:

```bash
cd /mnt/dev/work/pi05_TO_hybrid/openpi

export HF_LEROBOT_HOME=/mnt/dev/work/pi05_TO_hybrid/data
export OPENPI_DATA_HOME=/mnt/dev/work/pi05_TO_hybrid/cache/openpi
export WANDB_MODE=disabled
export XLA_FLAGS='--xla_gpu_enable_command_buffer='
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/train.py \
  pi05_rby1_randomized_pick_place_16d_lora \
  --exp-name rby1_randomized_pick_place_16d_30k_xla_retry_20260923 \
  --num-workers 8 \
  --no-wandb-enabled \
  2>&1 | tee \
  /mnt/dev/work/pi05_TO_hybrid/logs/rby1_16d_train_30k_xla_retry_20260923.log
```

`XLA_FLAGS='--xla_gpu_enable_command_buffer='`를 빠뜨리지 않는다. 첫 실행은 data loader와 base checkpoint 복원까지 성공했지만 step 0 직후 다음 오류로 종료되었다.

```text
CUDA_ERROR_INVALID_VALUE: Failed to set memcpy d2d node params
```

Command buffer를 비활성화한 retry는 정상적으로 여러 step을 진행했다. 새 retry를 만들 때는 이미 존재하는 experiment directory를 무조건 덮어쓰지 말고 새 exp name을 사용하거나 checkpoint resume 의도를 명확히 확인한다.

### 12.3 모니터링과 완료 판정

```bash
tmux ls
tmux attach -t rby1_16d_train
pgrep -af 'scripts/train.py'
tail -f /mnt/dev/work/pi05_TO_hybrid/logs/rby1_16d_train_30k_xla_retry_20260923.log
```

Checkpoint root:

```text
/mnt/dev/work/pi05_TO_hybrid/openpi/checkpoints/
  pi05_rby1_randomized_pick_place_16d_lora/
  rby1_randomized_pick_place_16d_30k_xla_retry_20260923/
```

완료는 다음을 모두 확인해야 한다.

1. `scripts/train.py` process가 정상 종료했다.
2. 로그가 traceback 없이 30,000 steps 완료를 기록한다.
3. 최종 checkpoint 디렉터리가 존재하고 임시 Orbax 디렉터리가 아니다.
4. Checkpoint assets 아래에 대응 `norm_stats.json`이 포함되었는지 확인한다.
5. 최소한 한 batch inference/load smoke test를 실행한다.

tmux session이 사라졌다는 사실만으로 정상 완료라고 판단하면 안 된다.

## 13. Timestamp/MP4 PTS 사고와 이번 해결

이 항목은 데이터 수집 시 가장 중요한 회귀 방지 사항이다.

### 13.1 과거 근본 원인

과거 데이터셋은 Parquet timestamp와 MP4 PTS가 서로 다른 시간축을 사용했다.

MP4는 정확한 15 FPS CFR(Constant Frame Rate)로 저장되므로 frame `i`의 PTS는 다음과 같다.

```text
video_pts[i] = i / 15
             = 0.0000, 0.0666667, 0.1333333, ...
```

반면 `rby1_transport_14d`의 Parquet는 MuJoCo physics step 시간을 기록했다.

```text
Parquet: 0.066, 0.132, 0.198, ...
MP4:     0.000, 0.0666667, 0.1333333, ...
```

MuJoCo timestep이 0.002초이고, 15 Hz에 가까운 capture 간격을 정수 step으로 반올림하면 33 steps가 된다.

```text
33 × 0.002 = 0.066 seconds
```

그러나 MP4의 한 frame 간격은 정확히 `1/15 = 0.066666... seconds`다. 따라서 episode가 길어질수록 timestamp 오차가 누적됐다. 긴 episode 끝에서는 약 0.52초, 약 8 frames까지 차이가 날 수 있었다.

더 오래된 `rby1_dataset_v1`은 reset 후 `settle_scene(..., 1.5초)`를 수행하고 절대 `data.time`을 저장했기 때문에 다음처럼 약 1.5초 offset도 포함했다.

```text
원본 timestamp: 1.502, 1.570, 1.638, ...
영상 PTS:       0.000, 0.0667, 0.1333, ...
```

Parquet row `i`, 세 camera frame `i`, state `i`, action `i`의 **순서 자체는 맞았다**. 잘못된 것은 timestamp label이었다. 하지만 LeRobot은 timestamp로 video frame을 조회하므로 이 불일치를 무시하면 다른 영상 frame과 state/action이 결합될 수 있다.

실제로 normalization 단계의 `check_timestamps_sync`가 다음 오류로 중단했다.

```text
ValueError: One or several timestamps unexpectedly violate the tolerance
diff: 0.066
timestamps: [0.066, 0.132]
```

확인 가능한 과거 기록상, 잘못된 데이터로 학습이 끝난 것이 아니라 LeRobot normalization/data loading 검증 단계에서 학습 준비가 실패한 문제였다.

### 13.2 과거 보정 과정의 2차 문제

당시 timestamp patch script 초기 버전은 각 Parquet 옆에 `*.parquet.bak`을 만들었다. Hugging Face/LeRobot dataset builder가 이 파일까지 shard로 읽으면서 dataset이 정확히 두 배가 되었다.

```text
정상 데이터:       609,094 frames
백업 포함 데이터: 1,218,188 frames
```

백업에는 보정 전 0.066초 timestamp가 남아 있었기 때문에 정상 Parquet를 수정했어도 normalization이 다시 실패했다.

과거 해결 방법은 state/action, MP4, frame 순서를 그대로 유지하고 Parquet timestamp만 다음처럼 다시 기록하는 것이었다.

```text
timestamp[i] = frame_index[i] / 15
```

현재 `src/scripts/patch_dataset_timestamps.py`는 backup을 dataset 밖에 만들지만, 이번 randomized dataset에는 이 patch를 실행할 필요가 없다.

### 13.3 이번 데이터셋에서 재발하지 않은 이유

이번 collector는 처음부터 raw simulator time을 timestamp로 저장하지 않는다. `EpisodeRecorder.on_step()`이 각 frame을 append할 때 다음 값을 생성한다.

```python
timestamp = len(episode) / fps
frame_index = len(episode)
```

Writer는 이 timestamp를 Parquet에 기록하고, 같은 `EpisodeBuffer`의 camera frame들을 `fps=15`로 MP4에 인코딩한다.

```text
Parquet timestamp[k] = k / 15
MP4 PTS[k]           = k / 15
```

즉 이번에는 보정 script로 사후 수정한 것이 아니라 **수집 시점부터 Parquet와 MP4가 같은 nominal media timeline을 사용**한다. MuJoCo capture가 33 physics steps마다 일어나더라도 dataset timestamp label은 MP4와 동일한 15 FPS grid다. State/action/image는 한 번의 recorder callback에서 같은 row/frame index로 묶인다.

실제 episode 0 검사 결과:

```text
Parquet timestamp:
0.000000, 0.06666667, 0.13333334, 0.20000000, ...

MP4 PTS:
0.000000, 0.066667, 0.133333, 0.200000, ...

Parquet rows: 311
MP4 frames:   311
MP4 FPS:      15/1
```

Float32 표현 차이는 validator 허용오차보다 충분히 작다. Randomized dataset validator는 모든 2,000 Parquet가 `arange(n) / 15`와 `atol=2e-6` 안에서 같은지 검사했고, 6,000 MP4를 모두 decode하여 episode row 수와 frame 수가 같은지 확인했다.

또한 서버의 실제 LeRobot/OpenPI loader가 세 camera와 action horizon을 읽어 batch를 만들고 normalization을 완료했다. Timestamp와 PTS가 어긋났다면 과거와 같은 nearest-frame/timestamp sync 오류가 이 단계에서 다시 발생했어야 한다.

이 해결은 이번 randomized collector가 사용하는 `EpisodeRecorder` 경로에 대한 것이다. `collect_batch.py`, 과거 handoff/block-pick처럼 `timestamp=float(t_sim)`을 직접 쓰는 다른 legacy collector까지 자동으로 고쳐지는 것은 아니다.

### 13.4 Timestamp 회귀 방지 체크리스트

- Dataset timestamp는 `frame_index / fps`로 만든다. `data.time`을 직접 저장하지 않는다.
- 모든 camera, state, action은 같은 logical frame index에서 append한다.
- MP4는 metadata FPS와 같은 CFR로 인코딩한다.
- 전 episode의 Parquet timestamp grid를 검사한다.
- 전 MP4를 decode하여 Parquet row 수와 frame 수를 비교한다.
- 대표 episode는 `ffprobe`로 실제 PTS도 확인한다.
- Dataset 내부에 `*.parquet.bak`, `*.tmp`, 복구 shard를 두지 않는다.
- Timestamp 문제를 tolerance 완화로 우회하지 않는다.
- 새로운 collector는 반드시 실제 LeRobot video loader smoke test까지 통과시킨다.

대표 MP4 PTS 확인 예시:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=avg_frame_rate,r_frame_rate,time_base,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  datasets/rby1_randomized_pick_place_16d_v1/videos/chunk-000/\
observation.images.cam_high/episode_000000.mp4

ffprobe -v error -select_streams v:0 \
  -show_entries frame=best_effort_timestamp_time -of csv=p=0 \
  datasets/rby1_randomized_pick_place_16d_v1/videos/chunk-000/\
observation.images.cam_high/episode_000000.mp4 | head
```

## 14. 운영 주의사항

### 수집 전

- Randomization config, ready pose, planner를 바꾸면 100-reset preflight와 smoke를 다시 수행한다.
- `passed=true`, config fingerprint, scatter plot, rejection reason을 모두 확인한다.
- 기존 dataset directory를 다른 config로 재사용하지 않는다.
- 과일별 실제 성공 rollout을 눈으로 확인한다. Reset-only preflight만으로 grasp 품질은 보장되지 않는다.

### 수집 중

- 장시간 작업은 tmux와 `tee` log로 실행한다.
- 하나의 dataset에 collector 두 개를 동시에 쓰지 않는다.
- 실패 rollout을 `--save-failed`로 본 dataset에 섞지 않는다.
- 성공률뿐 아니라 failure reason 분포를 본다. 특정 과일에 실패가 몰리면 범위를 무작정 줄이기 전에 grasp geometry를 검사한다.
- Resume 전에 process가 정말 종료됐는지 확인한다.
- Disk 여유와 MP4 생성 실패를 주기적으로 확인한다.

### 수집 후

- `plan_index` 순서, split 연속성, fruit/arm/prompt 균형을 검사한다.
- Parquet 개수만 세지 말고 모든 MP4를 실제 decode한다.
- Timestamp grid와 대표 MP4 PTS를 확인한다.
- Dataset root 안에 backup/temp 파일을 두지 않는다.
- Reindex hard link의 공유 inode를 주의한다.
- 전송 전 dataset을 동결하고 validation 결과와 config fingerprint를 함께 보관한다.

### 전송/학습 전

- rsync는 재개 가능한 `--partial`을 사용하고 전송 후 checksum dry-run을 한다.
- 서버의 부모 repo와 OpenPI repo를 혼동하지 않는다.
- Dirty user file을 reset하지 않는다.
- `HF_LEROBOT_HOME`, repo ID symlink, norm stats asset ID가 같은 dataset을 가리키는지 확인한다.
- Validation/test를 normalization이나 training episode에 포함하지 않는다.
- Checkpoint와 `norm_stats.json`, config commit, dataset fingerprint를 함께 보관한다.
- Training을 중복 실행하지 말고 tmux/process/checkpoint directory를 먼저 확인한다.
- 이 서버에서는 XLA command buffer 비활성화 설정을 유지한다.

## 15. 최종 인수 체크리스트

```text
[ ] randomization config fingerprint 기록
[ ] 100/100 valid reset preflight 통과
[ ] 네 과일 smoke rollout 성공
[ ] 2,000 successful episodes
[ ] fruit 500개씩, arm 1,000개씩
[ ] split 1600/200/200 및 plan 순서 확인
[ ] 2,000 Parquet / 6,000 MP4
[ ] 621,892 total frames
[ ] 모든 timestamp = frame_index / 15
[ ] 모든 MP4 decode 및 row/frame count 일치
[ ] dataset 내부 *.bak / *.tmp 없음
[ ] rsync checksum dry-run 무출력
[ ] 서버 LeRobot/OpenPI loader batch 생성 성공
[ ] train 0..1599만으로 16-D norm_stats 생성
[ ] XLA command buffer 비활성화 후 30K training 실행
[ ] final checkpoint와 norm_stats/config/dataset fingerprint 보관
```
