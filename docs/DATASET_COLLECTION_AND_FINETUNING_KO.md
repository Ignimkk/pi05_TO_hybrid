# RBY1 데이터셋 수집 및 π0.5 Fine-tuning

> 이 문서는 기존 색상 블록 데이터셋 `rby1_dataset_v1`을 설명한다. 과일 적재 및
> 상자 들기 데이터셋 `rby1_transport_14d`의 실제 1,200-episode 수집, GPU 서버
> 전송, timestamp 보정, normalization 및 H200 학습 절차는
> [`RBY1_TRANSPORT_14D_DATA_PIPELINE_KO.md`](RBY1_TRANSPORT_14D_DATA_PIPELINE_KO.md)를
> 참고한다.

## 1. 개요

본 프로젝트는 RBY1 양팔 로봇이 색깔 블록을 집어 갈색 상자에 넣는 작업을
수행하도록 Vision-Language-Action(VLA) 모델을 학습하는 것을 목표로 한다.
학습에는 RBY1 MuJoCo 시뮬레이션에서 자동 생성한 시연 데이터를 사용하며,
데이터는 LeRobot ALOHA 호환 형식으로 저장한다.

전체 파이프라인은 다음과 같다.

```text
MuJoCo 장면 및 물체 위치 무작위화
    → scripted waypoint 및 IK로 전문가 궤적 생성
    → 성공 여부 판정
    → 성공 에피소드의 영상·상태·제어 명령 저장
    → LeRobot 데이터 검증 및 정규화 통계 계산
    → π0.5 base checkpoint에서 LoRA fine-tuning
    → 50-step action chunk 예측
    → MuJoCo 환경에서 closed-loop 평가
```

중요한 점은 본 데이터가 사람이 원격조작한 데이터가 아니라는 것이다. 목표
위치와 grasp/handoff waypoint를 정의하고, IK로 계산한 관절 목표를 부드럽게
추종시켜 생성한 자동화된 전문가 시연 데이터이다.

---

## 2. `data/` 디렉터리 구성

```text
data/
├── rby1_dataset_v1/       # 실제 fine-tuning에 사용하는 본 학습 데이터
├── rby1_dataset_smoke/    # 데이터 로더와 파이프라인 검증용 소규모 데이터
└── policy_records/        # 학습된 정책의 추론 입력·출력 기록
```

| 디렉터리 | 규모 | 용도 |
|---|---:|---|
| `rby1_dataset_v1` | 약 2.4 GB, 1,200 episodes | π0.5 LoRA fine-tuning |
| `rby1_dataset_smoke` | 약 48 MB, 26 episodes | 로딩·포맷·실행 smoke test |
| `policy_records` | 약 50 MB, 114 records | 추론 지연 및 action chunk 분석 |

`policy_records`는 학습 데이터가 아니다. 정책 실행 시의 관측과 모델 출력
action chunk를 기록한 평가·디버깅 자료이다.

---

## 3. 본 학습 데이터셋

### 3.1 기본 통계

`rby1_dataset_v1`의 구성은 다음과 같다.

| 항목 | 값 |
|---|---:|
| Robot | RBY1 |
| 데이터 형식 | LeRobot v2.0, ALOHA-compatible schema |
| 에피소드 수 | 1,200 |
| 전체 프레임 수 | 428,400 |
| FPS | 15 |
| 전체 기록 시간 | 28,560초, 약 7시간 56분 |
| 언어 태스크 수 | 6 |
| 카메라 수 | 3 |
| State 차원 | 14 |
| Action 차원 | 14 |
| 학습 split | `0:1200`, 전체 데이터 |
| Validation/Test split | 별도 구성 없음 |

본 데이터셋에는 성공한 에피소드 1,200개만 들어 있다. 수집 과정에서는 총
1,268회 실행하여 68회 실패했고, 실패 실행은 본 데이터셋에 저장하지 않았다.
따라서 수집 성공률은 약 94.6%이다.

### 3.2 언어 태스크

색상 3종과 작업을 마무리하는 손 2종을 조합한 6개 명령을 사용한다.

```text
put the red block in the brown box with your left hand
put the red block in the brown box with your right hand
put the green block in the brown box with your left hand
put the green block in the brown box with your right hand
put the blue block in the brown box with your left hand
put the blue block in the brown box with your right hand
```

각 언어 태스크에는 200개 에피소드가 있어 색상과 손에 따른 표본 수가
균등하다. 다만 같은 언어 명령 안에는 서로 다른 두 종류의 운동 궤적이
포함된다.

| 최종 작업 손 | 물체가 같은 쪽에 있을 때 | 물체가 반대쪽에 있을 때 |
|---|---|---|
| 왼손 | 왼손 single-arm pick-and-place 100개 | 오른손에서 왼손으로 handoff 100개 |
| 오른손 | 오른손 single-arm pick-and-place 100개 | 왼손에서 오른손으로 handoff 100개 |

따라서 “with your left hand”는 항상 왼손만 움직이라는 뜻이 아니라, 왼손이
최종적으로 상자에 넣는다는 의미이다. 물체가 오른쪽에 있으면 오른손이 먼저
집은 뒤 왼손에 전달한다. 오른손 명령도 같은 방식으로 구성된다.

### 3.3 에피소드 길이

| 동작 종류 | 프레임 | 시간 | 에피소드 수 |
|---|---:|---:|---:|
| Single-arm | 228 | 약 15.2초 | 600 |
| Handoff | 486 | 약 32.4초 | 600 |
| 전체 평균 | 357 | 약 23.8초 | 1,200 |

Handoff에는 한 팔의 접근·grasp뿐 아니라 두 팔 정렬, 물체 전달, 반대쪽 팔의
배치 및 복귀 동작이 포함되므로 single-arm보다 길다.

---

## 4. 프레임별 데이터 구성

각 에피소드는 영상과 비영상 데이터로 나누어 저장된다.

```text
rby1_dataset_v1/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── stats.json
├── data/chunk-000/
│   └── episode_XXXXXX.parquet
└── videos/chunk-000/
    ├── observation.images.cam_high/episode_XXXXXX.mp4
    ├── observation.images.cam_left_wrist/episode_XXXXXX.mp4
    └── observation.images.cam_right_wrist/episode_XXXXXX.mp4
```

현재 writer 구현은 1,200개 에피소드를 모두 `chunk-000`에 저장한다.

### 4.1 영상 관측

| Dataset key | MuJoCo camera | 역할 |
|---|---|---|
| `observation.images.cam_high` | `zed_left` | 장면 전체를 보는 외부 시점 |
| `observation.images.cam_left_wrist` | `wrist_cam_l` | 왼쪽 손목 시점 |
| `observation.images.cam_right_wrist` | `wrist_cam_r` | 오른쪽 손목 시점 |

세 영상 모두 RGB `224×224×3`, 15 FPS이며 H.264 MP4로 저장한다. 외부
카메라는 물체와 상자의 전역 관계를 제공하고, 손목 카메라는 grasp와 handoff
시의 근거리 시각 정보를 제공한다.

### 4.2 로봇 상태

`observation.state`는 현재 측정된 관절 상태를 나타내는 float32 14차원
벡터이다.

```text
[
  left_arm_0, ..., left_arm_5, left_gripper,
  right_arm_0, ..., right_arm_5, right_gripper
]
```

RBY1의 각 팔은 실제로 7-DoF이지만 ALOHA 형식과 맞추기 위해 마지막 wrist
관절 `arm_6`은 제외한다. 그리퍼는 수집 시 사용한 open 값으로 나누어
정규화하며, 기본 의미는 `0=closed`, `1=open`이다.

### 4.3 행동 레이블

`action` 역시 float32 14차원이며 state와 같은 순서를 사용한다. 관절 action은
프레임을 촬영한 시점에 MuJoCo position actuator로 보낸 절대 관절 목표값이고,
그리퍼 action은 정규화된 절대 개폐 명령이다.

즉 원본 Parquet에 저장된 action은 joint velocity나 delta가 아니다.

```text
observation.state[t] = 시점 t에서 측정한 실제 관절 상태
action[t]            = 시점 t에서 actuator에 적용한 절대 관절 목표
```

### 4.4 부가 필드

| 필드 | 설명 |
|---|---|
| `timestamp` | 에피소드 시작 기준 시간 |
| `frame_index` | 에피소드 내부 프레임 번호 |
| `episode_index` | 에피소드 번호 |
| `task_index` | `tasks.jsonl`의 언어 태스크 번호 |
| `index` | 에피소드 내부 순차 인덱스 |
| `next.done` | 마지막 프레임에서만 `true` |
| `next.reward` | 모든 프레임에서 0 |

Reward가 모두 0이므로 강화학습용 transition 데이터가 아니라, 관측에서 전문가
행동을 예측하는 behavior cloning/VLA fine-tuning 데이터로 보아야 한다.

---

## 5. 데이터 수집 방식

### 5.1 장면 초기화와 무작위화

각 실행은 MuJoCo 장면을 초기화하고 물체가 테이블에 안정적으로 놓일 때까지
physics를 진행한다. 이후 seed를 사용해 빨강·초록·파랑 블록 위치를 각 팔의
도달 가능 영역에서 무작위로 샘플링한다. 블록 사이에는 최소 간격을 두어
겹침을 방지한다.

색상별로 다음 네 가지 구성을 각각 100회 성공할 때까지 수집한다.

| 구성 | Spawn side | 수행 방식 | 최종 작업 손 |
|---|---|---|---|
| `left_single` | 왼쪽 | 왼손 단독 | 왼손 |
| `left_handoff` | 왼쪽 | 왼손 → 오른손 전달 | 오른손 |
| `right_handoff` | 오른쪽 | 오른손 → 왼손 전달 | 왼손 |
| `right_single` | 오른쪽 | 오른손 단독 | 오른손 |

3 colors × 4 configurations × 100 successes로 총 1,200개가 된다.

### 5.2 전문가 궤적 생성

Single-arm 시나리오는 대략 다음 순서로 수행된다.

```text
rest
→ block 위 pre-approach
→ grasp 위치로 하강
→ adaptive gripper close
→ block lift
→ 상자 위로 이동
→ release
→ retract
→ rest 복귀 및 안정화
```

Handoff 시나리오에는 다음 과정이 추가된다.

```text
첫 번째 팔이 물체 grasp
→ 두 팔의 end-effector를 전달 위치에 정렬
→ 받는 팔의 gripper close
→ 주는 팔의 gripper open
→ 받는 팔이 상자로 이동해 release
```

각 end-effector 목표 pose는 waypoint로 정의한다. `mink` 기반 masked
kinematic IK로 목표 관절 위치를 계산하고, 현재 명령에서 다음 관절 목표까지
joint-space interpolation하여 급격한 움직임을 줄인다. Grasp에는 물체 접촉에
따라 닫힘 정도를 조절하는 adaptive close도 사용한다.

### 5.3 프레임 기록

시뮬레이션이 진행되는 동안 15 Hz 주기로 다음 항목을 동시에 기록한다.

- 외부 카메라 RGB
- 왼쪽·오른쪽 손목 카메라 RGB
- 현재 관절 및 그리퍼 상태
- 현재 actuator 제어 목표
- timestamp와 frame index
- 언어 명령

Parquet의 행 `i`, 세 MP4의 영상 프레임 `i`, state/action `i`는 같은
시뮬레이션 시점을 나타낸다. 영상은 15 FPS CFR로 저장하므로 timestamp도
`frame_index / 15`에 맞춰 정렬되어야 한다.

### 5.4 성공 판정과 저장

시나리오가 끝나면 목표 블록이 갈색 상자의 xy 경계 안에 있고 상자 바닥보다
위에 있는지 검사한다.

- 성공: 에피소드의 Parquet, MP4 3개, metadata 저장
- 실패: 본 데이터셋에서는 저장하지 않고 다음 seed로 재시도
- Smoke 수집에서 `--save-failed` 사용 시: prompt 앞에 `[FAIL]`을 붙여 저장

수집 진행 상태는 `collect_dataset_stats.json`에 매 실행 후 기록하므로 중단된
수집을 이어서 수행할 수 있다.

---

## 6. 학습 전 데이터 준비

### 6.1 구조 검증

다음 명령으로 metadata 수치, 태스크 분포, 에피소드 길이, Parquet/MP4 누락,
state/action shape를 검사한다.

```bash
python scripts/validate_dataset.py --dataset data/rby1_dataset_v1
```

현재 본 데이터셋은 1,200 episodes, 428,400 frames, 6 tasks와 세 카메라
파일이 모두 일치한다.

### 6.2 원본 데이터 통계

`data/rby1_dataset_v1/meta/stats.json`에는 원본 14차원 state/action의
`min`, `max`, `mean`, `std`, `q01`, `q99`가 들어 있다.

```bash
python scripts/compute_stats.py --dataset data/rby1_dataset_v1
```

이 파일은 데이터 자체를 분석하고 검증하는 데 유용하다. 그러나 실제 OpenPI
학습에서는 아래의 OpenPI config 기반 통계도 별도로 계산해야 한다. Config
기반 통계는 absolute action을 joint delta로 변환한 이후의 값이기 때문이다.

### 6.3 LeRobot 경로 등록

학습 config의 repo ID는 `local/rby1_dataset_v1`이다. 따라서 사용하는
LeRobot 버전의 로컬 dataset home 아래에서 다음 경로로 보이도록 복사하거나
심볼릭 링크해야 한다.

```text
<HF_LEROBOT_HOME>/local/rby1_dataset_v1
```

경로가 올바른지는 normalization 계산 전에 반드시 확인한다. LeRobot 버전에
따라 환경변수 이름이 `HF_LEROBOT_HOME` 또는 `LEROBOT_HOME`일 수 있다.

### 6.4 OpenPI 학습용 정규화 통계

OpenPI 소스 루트에서 다음을 실행한다.

```bash
cd src/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_rby1_lora
```

결과는 기본 설정에서 다음 위치에 생성된다.

```text
assets/pi05_rby1_lora/local/rby1_dataset_v1/norm_stats.json
```

π0.5 config는 quantile normalization을 사용하므로 `q01`, `q99`가 필요하다.
이 통계는 학습 checkpoint의 asset으로 복사되어 추론 시 같은 역정규화에
사용된다.

---

## 7. 실제 학습 시 사용되는 데이터

디스크의 한 프레임이 모델 입력으로 전달되기까지 다음 변환을 거친다.

### 7.1 원본 feature repack

```text
observation.images.cam_high        → images.cam_high
observation.images.cam_left_wrist  → images.cam_left_wrist
observation.images.cam_right_wrist → images.cam_right_wrist
observation.state                  → state
action                             → actions
tasks.jsonl의 task                 → prompt
```

### 7.2 50-step action chunk 생성

π0.5 RBY1 config의 `action_horizon`은 기본값 50이다. 현재 시점 `t`의 관측
하나에 대해 loader가 다음과 같은 미래 행동 시퀀스를 만든다.

```text
input:
  images[t], state[t], prompt

target:
  action[t : t+50]
```

15 Hz 기준 50 step은 약 3.33초의 미래 궤적이다. 모델은 다음 한 동작만
예측하는 것이 아니라 양팔의 연속적인 action chunk 전체를 생성하도록
학습된다.

### 7.3 Absolute action에서 joint delta로 변환

원본 action은 절대 관절 목표지만 학습 직전에는 현재 state 기준 delta로
변환한다.

```text
joint_target_delta[t+k] = raw_action[t+k] - state[t]
```

변환 대상은 왼팔 6개와 오른팔 6개 관절이다. 두 gripper 차원은 absolute
값을 유지한다.

```text
학습 target 14차원:
[
  Δleft_joint_0 ... Δleft_joint_5, left_gripper_absolute,
  Δright_joint_0 ... Δright_joint_5, right_gripper_absolute
]
```

추론 후에는 예측 delta에 현재 state를 다시 더해 절대 관절 목표로 복원한다.

### 7.4 ALOHA 변환 설정

RBY1의 14차원 배열 순서는 ALOHA와 같지만 실제 Trossen ALOHA 하드웨어는
아니다. 따라서 `adapt_to_pi=False`로 설정하여 Trossen 전용 관절 부호 반전과
그리퍼 기구 변환을 적용하지 않는다.

### 7.5 정규화와 모델 입력 형태

π0.5에서는 변환된 state/action을 `q01`, `q99` 기반으로 정규화한다. 이후:

- 세 카메라를 `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`로 매핑
- 영상을 `224×224`로 맞춤
- 언어 prompt와 state를 π0.5 입력 token으로 변환
- 최대 token 길이 200 사용
- 14차원 state/action 뒤에 0을 붙여 모델 내부 규격인 32차원으로 padding

32차원 전체가 로봇 자유도라는 뜻은 아니다. 실제 유효한 로봇 출력은 앞의
14차원이며, 나머지 18차원은 π0.5 공통 action dimension을 맞추기 위한
padding이다.

최종 학습 샘플의 개념적 형태는 다음과 같다.

```text
Observation
├── base_0_rgb:        224×224×3
├── left_wrist_0_rgb:  224×224×3
├── right_wrist_0_rgb: 224×224×3
├── state:             32       # 실제 14 + zero padding 18
└── prompt/state tokens

Target actions
└── 50×32                      # 실제 50×14 + zero padding
```

---

## 8. π0.5 LoRA Fine-tuning

### 8.1 초기 모델

학습은 공개된 π0.5 base parameter에서 시작한다.

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

사전학습된 vision-language-action 표현을 유지하면서 RBY1의 카메라 시점,
관절 공간, 그리퍼 표현 및 작업 궤적에 적응시키기 위해 full fine-tuning 대신
LoRA를 사용한다.

### 8.2 LoRA 구성

RBY1 config는 두 Gemma 계열 모듈에 LoRA를 적용한다.

| 구성요소 | Variant | LoRA rank | alpha |
|---|---|---:|---:|
| PaliGemma language/vision backbone | `gemma_2b_lora` | 16 | 16 |
| Action expert | `gemma_300m_lora` | 32 | 32 |

Attention과 feed-forward layer에 저랭크 adapter를 추가하고, 기존 해당
backbone weight는 freeze한다. 따라서 전체 base model을 다시 학습하는 것보다
필요 메모리와 학습 가능한 parameter 수를 줄일 수 있다.

### 8.3 학습 목적함수

π0.5는 action chunk에 noise를 섞은 중간 상태를 만들고, noise에서 실제 action
분포로 이동하는 velocity field를 예측하는 flow-matching 방식으로 학습한다.
손실은 50-step action chunk 전반에서 예측 velocity와 정답 velocity 사이의
평균 제곱 오차이다.

즉 일반적인 “다음 관절값 하나에 대한 MSE”가 아니라, 영상·언어·현재 상태를
조건으로 미래 양팔 궤적 전체의 생성 과정을 학습한다.

### 8.4 현재 학습 설정

실제 `pi05_rby1_lora` config는 다음과 같다.

| 항목 | 설정 |
|---|---:|
| Batch size | 32 |
| Train steps | 30,000 |
| 총 sample draws | 960,000 |
| Warmup | 1,000 steps |
| Peak learning rate | `5e-5` |
| Final learning rate | `5e-6` |
| Schedule | Cosine decay |
| Optimizer | AdamW |
| Gradient clipping | global norm 1.0 |
| EMA | 사용하지 않음 |
| Log interval | 100 steps |
| Checkpoint interval | 5,000 steps |
| Seed | 42 |

428,400 frame samples를 기준으로 단순 환산하면 960,000 sample draws는 약
2.24 dataset passes에 해당한다. 실제 loader의 episode 끝 action chunk 처리와
shuffle 방식에 따라 체감 epoch는 조금 달라질 수 있다.

### 8.5 학습 실행

OpenPI 소스 루트에서 normalization 통계를 먼저 계산한 뒤 학습한다.

```bash
cd src/openpi

uv run scripts/compute_norm_stats.py \
    --config-name pi05_rby1_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_rby1_lora \
    --exp-name=full_run_30k \
    --overwrite
```

Checkpoint는 기본적으로 다음 위치에 저장된다.

```text
checkpoints/pi05_rby1_lora/full_run_30k/
├── 4999/
├── 9999/
├── 14999/
├── 19999/
├── 24999/
└── 29999/
```

동일한 experiment를 이어서 학습할 때는 기존 결과를 덮어쓰지 말고 config의
resume 옵션을 사용해야 한다.

---

## 9. Fine-tuning 이후 추론

추론 시에도 학습 때와 동일한 세 카메라, 14차원 state, prompt, 정규화 통계와
입출력 변환을 사용해야 한다.

모델은 한 번 호출할 때 `(50, 14)`에 해당하는 유효 action chunk를 반환한다.
프로젝트의 기본 open-loop 실행 길이는 앞의 8 step이다.

```text
현재 관측 전송
→ 50-step action chunk 생성
→ 앞 8 step 실행
→ 새 영상과 state 측정
→ 다시 정책 호출
```

15 Hz에서 8 step은 약 0.53초이다. 50 step 전체를 open-loop로 실행하지 않고
8 step마다 다시 관측하는 이유는 물체 접촉이나 tracking error에 대응하기
위해서다.

추론 시 주의할 사항은 다음과 같다.

- 학습과 동일한 관절 순서 및 gripper normalization을 사용한다.
- Prompt는 가능하면 학습에 사용한 6개 문장과 정확히 맞춘다.
- `adapt_to_pi=False`를 유지한다.
- Checkpoint에 포함된 RBY1 normalization asset을 사용한다.
- 출력의 앞 14차원만 로봇 명령으로 사용한다.
- Action delta를 현재 state 기준 절대 목표로 복원한 뒤 actuator에 적용한다.

`policy_records/step_N.npy`에는 이러한 추론의 입력과 50-step 출력, 추론 시간이
기록되어 있으며 action chunk 내부 움직임과 chunk 경계 불연속을 분석하는 데
사용한다.

---

## 10. Smoke 데이터셋과 평가상 한계

`rby1_dataset_smoke`는 26 episodes, 8,508 frames로 구성되며 성공 25개와
실패 1개를 포함한다. 실패 prompt에는 `[FAIL]` 접두사가 있다. 이 데이터는
본 학습용이 아니라 다음 항목을 빠르게 확인하기 위한 것이다.

- LeRobot dataset 로딩 가능 여부
- Parquet와 영상의 frame alignment
- 카메라 key와 state/action shape
- 실패 에피소드 표기와 필터링
- 작은 batch의 학습 코드 실행 여부

현재 데이터와 실험 설계에는 다음 한계가 있다.

1. 학습 split만 있고 독립적인 validation/test demonstration split이 없다.
2. 전문 시연이 scripted IK에서 생성되어 사람의 다양한 회복 행동이 부족하다.
3. 본 학습 데이터에는 실패와 실패 회복 과정이 없다.
4. 태스크 문장이 6개로 고정되어 언어 표현의 다양성이 낮다.
5. Single-arm과 handoff가 같은 prompt 아래 있으므로 spawn 위치가 어느 수행
   방식을 선택할지 결정하는 핵심 시각 단서가 된다.
6. 시뮬레이션 데이터이므로 실제 RBY1에 적용하려면 카메라 calibration,
   dynamics, latency, gripper 및 관절 표현 차이를 추가로 검증해야 한다.

향후에는 독립 seed 기반 평가 split, 물체·상자 위치 범위 확대, 조명과 카메라
domain randomization, 실패 복구 시연, prompt paraphrase, 실제 로봇 데이터
혼합을 통해 일반화 성능을 개선할 수 있다.

---

## 11. 관련 구현

| 파일 | 역할 |
|---|---|
| `rby1_manipulation.data.collect_dataset` | 12개 수집 구성 실행 및 재시도 |
| `rby1_manipulation.tasks.block_pick` | Single-arm 시연 생성 |
| `rby1_manipulation.tasks.handoff_left_to_right` | 왼손→오른손 handoff |
| `rby1_manipulation.tasks.handoff_right_to_left` | 오른손→왼손 handoff |
| `rby1_manipulation.data.episode` | LeRobot Parquet/MP4 writer |
| `scripts/validate_dataset.py` | 데이터셋 구조·분포 검증 |
| `scripts/compute_stats.py` | 원본 dataset 통계 생성 |
| `scripts/patch_dataset_timestamps.py` | Parquet timestamp와 영상 PTS 정렬 |
| `src/openpi/src/openpi/training/config.py` | `pi05_rby1_lora` 학습 설정 |
| `src/rby1_bringup/pi05_infer.py` | Fine-tuned 정책의 MuJoCo 추론 |
| `scripts/inspect_policy_records.py` | 추론 기록 및 action chunk 분석 |
