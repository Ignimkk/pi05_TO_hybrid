# RBY1 과일 운반 14-D 데이터 수집·전송·전처리·학습

## 1. 문서 목적과 현재 상태

이 문서는 RBY1용 알고리즘을 MuJoCo에서 검증한 뒤 실제 로봇에 적용하기 위한
sim-to-real 작업 중, **과일을 상자에 적재하고 상자를 들어 올리는 VLA 데이터의
생성부터 π0.5 LoRA 학습까지** 실제 수행한 절차를 재현 가능하게 정리한다.

이 문서는 기존 색상 블록 데이터셋 `rby1_dataset_v1` 문서와 구분된다. 이번
데이터의 실제 이름은 `rby1_transport_14d`이며, 기존 OpenPI config의 repo ID를
재사용하기 위해 GPU 서버에서만 `local/rby1_dataset_v1`이라는 심볼릭 링크
별칭으로 노출한다.

2026-08-13 기준 상태는 다음과 같다.

| 단계 | 상태 |
|---|---|
| MuJoCo 수집 | 완료, 성공 1,200/1,200 episode |
| 로컬 데이터 검증 | 완료 |
| GPU 서버 전송 | 완료 |
| 서버 `meta/stats.json` | 완료, 609,094 frame 집계 |
| timestamp 보정 | 완료, 1,200 Parquet |
| OpenPI `norm_stats.json` | H200 서버에서 계산 중 |
| π0.5 LoRA 30K 학습 | norm 계산 검증 성공 후 자동 시작하도록 구성 |

전체 흐름은 다음과 같다.

```text
MuJoCo expert rollout
    → 성공 episode만 LeRobot 14-D 형식으로 저장
    → 로컬에서 1,200 episode 구조·영상 검증
    → bastion을 경유해 GPU 서버로 rsync
    → 서버에서 multi-chunk 구조 검증
    → raw dataset stats 계산
    → Parquet timestamp를 MP4의 15 FPS PTS와 정렬
    → LeRobot/OpenPI normalization stats 계산
    → π0.5 base checkpoint에서 LoRA 30K fine-tuning
    → checkpoint와 norm stats를 함께 사용해 추론·평가
```

---

## 2. 인프라와 경로

### 2.1 로컬 PC

```text
OS                  Ubuntu
workspace           /home/mk/dev_ws/vla/pi0_TO_ws
source repository   /home/mk/dev_ws/vla/pi0_TO_ws/src
dataset             /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_transport_14d
Python              /home/mk/venv/pi0_TO_env/bin/python
```

수집은 GPU가 없는 로컬 환경의 MuJoCo OSMesa software renderer로 수행했다.

### 2.2 GPU 서버

```text
GPU                 NVIDIA H200, 141 GB HBM3
bastion             blunex@ai.amrc.kr:21151
container endpoint  root@172.21.121.112:31905
workspace           /mnt/dev/work/pi05_TO_hybrid
OpenPI              /mnt/dev/work/pi05_TO_hybrid/openpi
Python              /mnt/dev/work/pi05_TO_hybrid/openpi/.venv/bin/python
dataset             /mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d
logs                /mnt/dev/work/pi05_TO_hybrid/logs
checkpoints         /mnt/dev/work/pi05_TO_hybrid/checkpoints
```

SSH 연결은 다음과 같은 2-hop 구조이다.

```text
local PC
    → bastion ai.amrc.kr:21151
    → K8s/Docker container NodePort 172.21.121.112:31905
```

---

## 3. 데이터셋의 과업 범위

이번 데이터셋은 `transport_load_and_carry`의 과일 pick 및 양팔 crate grasp를
기반으로 하지만 다음 동작은 포함하지 않는다.

- wheel 또는 mobile base 이동
- 선반까지 운반 및 선반 위 배치
- 정적·동적 장애물 회피

장애물 회피는 학습 데이터에 포함하지 않고 이후 별도 알고리즘으로 평가한다.

언어 지시는 세 계열로 구성했다.

| 계열 | 의미 | episode 수 |
|---|---|---:|
| `pack_only` | 지정된 과일 1~4개를 상자에 넣고 종료 | 400 |
| `lift_only` | 과일 0~4개가 미리 든 상자를 들어 올림 | 400 |
| `pack_and_lift` | 과일 1~4개를 넣은 뒤 상자를 들어 올림 | 400 |

세부 수량은 다음과 같다.

| 계열 | 과일 수별 분포 |
|---|---|
| `pack_only` | 1·2·3·4개 각각 100 episode |
| `lift_only` | 사전 적재 0·1·2·3·4개 각각 80 episode |
| `pack_and_lift` | 1·2·3·4개 각각 100 episode |

과일은 `apple`, `banana`, `orange`, `pear` 네 종류다. 노출 수 역시 균형을
맞췄다.

| 계열 | apple | banana | orange | pear |
|---|---:|---:|---:|---:|
| `pack_only` | 250 | 250 | 250 | 250 |
| `lift_only` 사전 적재 | 200 | 200 | 200 | 200 |
| `pack_and_lift` | 250 | 250 | 250 | 250 |

과일 조합에 따라 최종 언어 task 문자열은 총 31종이다. `lift_only`의 prompt는
사전 적재 수와 관계없이 `lift the crate`이므로 이 지시만 400 episode다.

---

## 4. 장면 다양성과 전문가 동작

### 4.1 Fruit grid

과일 위치는
`rby1_manipulation/src/rby1_manipulation/config/grids/fruit_grid.json`의 grid를
사용한다.

- 좌측 팔 영역 후보점 4개
- 우측 팔 영역 후보점 4개
- 같은 쪽 두 과일 간 최소 75 mm 확보
- 좌측 pair 4개 × 우측 pair 4개 = 16 layout
- 각 layout 75 episode
- 각 과일은 table slot 0·1·2·3에 각각 300회 등장
- 네 과일의 실행 순서는 24개 순열을 순환

`lift_only`에서 상자에 미리 들어간 과일을 제외한 나머지 과일도 table grid에
distractor로 남는다.

### 4.2 전문가 궤적

과일 하나를 상자에 넣는 기본 순서는 다음과 같다.

```text
object over
→ hover
→ descend
→ descend trim
→ adaptive gripper close
→ lift
→ carry
→ crate 안으로 lower
→ release
→ vertical retract
```

다음 과일을 반대쪽 팔이 담당하면 이전 팔을 초기 joint rest pose로 먼저
복귀시킨다. 이는 상자 위의 공유 작업공간에서 양팔이 충돌하는 문제를 막기 위한
에피소드 내부 안전 동작이다. 같은 팔이 연속 작업할 때는 불필요한 복귀를 생략한다.

상자 들기는 양팔을 handle 부근에 접근시킨 뒤 양쪽 그리퍼를 닫고 수직으로
들어 올리는 순서다.

### 4.3 수집 중 발견하고 수정한 실패 원인

| 문제 | 원인 | 적용한 개선 |
|---|---|---|
| banana pick 때 finger-table 충돌 | 낮고 휘어진 mesh 때문에 계산 grasp 높이가 너무 낮음 | banana 최소 grasp offset 22 mm |
| pear가 close 후 첫 lift에서 이탈 | 좁은 목 부분을 잡는 25.9 mm offset | 넓은 몸통을 잡도록 pear offset 15 mm |
| 왼팔 작업 후 오른팔 작업 때 충돌 | 이전 팔이 crate 위에 남음 | 팔이 바뀔 때 이전 팔을 rest pose로 복귀 |
| 숨어 있는 양팔 충돌 | 최종 성공 판정만으로 접촉을 놓칠 수 있음 | inter-arm contact를 100 Hz로 검사 |
| 과일 grasp 중 table 접촉 | mesh와 pose 조합에 따라 finger 접촉 가능 | finger-table contact를 100 Hz로 검사 |

table 또는 inter-arm 접촉이 한 번이라도 검출된 episode는 성공 데이터로 저장하지
않는다.

### 4.4 동작 속도

검증된 기본 `speed_scale`은 `1.25`다. `1.5`에서는 banana가 이동 중 이탈한
사례가 있어 전체 수집에는 적용하지 않았다. 작은 물체의 마지막 접근과 gripper
close 사이 대기도 지나치게 줄이면 후속 상자 grasp가 불안정해져 검증된 기본값을
유지했다.

---

## 5. 프레임과 LeRobot 데이터 구조

### 5.1 State/action 14-D

`observation.state`와 `action`은 모두 float32 14차원이며 순서는 동일하다.

```text
[
  left_arm_0, ..., left_arm_5, left_gripper,
  right_arm_0, ..., right_arm_5, right_gripper
]
```

- state: 현재 측정된 MuJoCo 관절 위치와 정규화된 gripper 상태
- action: 같은 시점의 position actuator 절대 목표와 gripper 절대 명령
- gripper: 대략 `0=closed`, `1=open`
- 원본 action은 velocity나 delta action이 아님

OpenPI 학습 config가 뒤 단계에서 12개 arm joint action만 현재 state 대비 delta로
변환하며, 두 gripper action은 absolute로 유지한다.

### 5.2 영상

| Dataset key | MuJoCo camera | 형식 |
|---|---|---|
| `observation.images.cam_high` | `zed_left` | RGB 224×224, H.264, 15 FPS |
| `observation.images.cam_left_wrist` | `wrist_cam_l` | RGB 224×224, H.264, 15 FPS |
| `observation.images.cam_right_wrist` | `wrist_cam_r` | RGB 224×224, H.264, 15 FPS |

Parquet 행 `i`와 세 영상의 frame `i`는 같은 시뮬레이션 상태를 나타낸다.

### 5.3 기타 필드

| 필드 | 의미 |
|---|---|
| `timestamp` | 에피소드 내 nominal video PTS, `frame_index / fps` |
| `frame_index` | 에피소드 내부 프레임 번호 |
| `episode_index` | 전체 데이터셋 episode 번호 |
| `task_index` | `tasks.jsonl`의 prompt 번호 |
| `index` | episode 내부 순차 인덱스 |
| `next.done` | 마지막 프레임만 `true` |
| `next.reward` | 현재 데이터에서는 0 |

### 5.4 실제 디렉터리 구조

LeRobot chunk 크기는 1,000 episode다. 따라서 1,200 episode는 두 chunk로
나뉜다.

```text
rby1_transport_14d/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── stats.json
├── data/
│   ├── chunk-000/       # episode 000000~000999
│   └── chunk-001/       # episode 001000~001199
├── videos/
│   ├── chunk-000/
│   │   ├── observation.images.cam_high/
│   │   ├── observation.images.cam_left_wrist/
│   │   └── observation.images.cam_right_wrist/
│   └── chunk-001/
│       ├── observation.images.cam_high/
│       ├── observation.images.cam_left_wrist/
│       └── observation.images.cam_right_wrist/
├── transport_collection_plan.json
├── transport_collection_stats.json
└── transport_episode_manifest.jsonl
```

Sidecar 의미는 다음과 같다.

- `transport_collection_plan.json`: 1,200개 균형 schedule과 seed
- `transport_collection_stats.json`: plan별 시도 횟수, 성공 여부, 마지막 오류
- `transport_episode_manifest.jsonl`: plan과 저장 dataset episode index의 대응

---

## 6. 로컬 데이터 수집

### 6.1 단일 episode 검증

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws/src
source /home/mk/venv/pi0_TO_env/bin/activate

# 과일만 넣기
python -m rby1_manipulation.tasks.transport_pack_lift \
  --task pack_only --objects apple banana --headless

# 과일이 미리 든 상자만 들기
python -m rby1_manipulation.tasks.transport_pack_lift \
  --task lift_only --preloaded apple orange pear --headless

# 과일을 넣고 상자 들기
python -m rby1_manipulation.tasks.transport_pack_lift \
  --task pack_and_lift --objects banana apple --headless
```

### 6.2 전체 schedule 확인

```bash
python -m rby1_manipulation.data.collect_transport_dataset \
  --output-dir /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_transport_14d \
  --episodes 1200 \
  --dry-run --dry-run-limit 12
```

### 6.3 전체 수집과 재개

실제 최종 재시도에는 다음 설정을 사용했다.

```bash
env LP_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  /home/mk/venv/pi0_TO_env/bin/python \
  -m rby1_manipulation.data.collect_transport_dataset \
  --output-dir /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_transport_14d \
  --episodes 1200 \
  --timeout 900 \
  --max-attempts 10 \
  --speed-scale 1.25
```

수집기를 중단한 뒤 같은 command를 실행하면 stats를 읽고 이미 성공한 plan은
건너뛴다. 기본 5회 시도 후 남은 실패 plan은 pear grasp와 팔 교대 로직을 수정한
후 `--max-attempts 10`으로 재개했다. 마지막 재개 구간은 1,161개에서 시작해
남은 39개를 추가하여 1,200개를 완성했으며 이 재개 실행은 약 1.09시간이었다.

### 6.4 software rendering 성능 개선

초기 구현은 simulation step 중 15 Hz마다 세 카메라를 순차 렌더링하여 매우
느렸다. 다음과 같이 변경했다.

```text
이전:
physics 진행 → cam_high → left wrist → right wrist → 다음 physics

현재:
physics 진행 중 state/action/qpos만 기록
→ 성공 판정
→ 세 독립 process가 qpos trajectory를 카메라별로 병렬 렌더링
→ Parquet와 MP4 저장
```

추가 최적화:

- OSMesa policy render의 reflection 비활성화
- 실패하여 저장하지 않는 episode는 policy camera 렌더링 생략
- 세 camera worker에 CPU thread 사용
- 영상 해상도, camera pose, shadow, 15 FPS, state/action 의미는 유지

실측 예시:

- policy camera microbenchmark: reflection 비활성화 후 약 2.2배 개선
- 단일 pear episode: 243 frame, 영상 3개 저장까지 31.47초
- 4-fruit episode: 기존 약 200초 사례에서 117초 사례로 단축

### 6.5 최종 로컬 검증 결과

| 항목 | 결과 |
|---|---:|
| 성공 episode | 1,200 |
| 실패로 저장된 episode | 0 |
| 전체 frame | 609,094 |
| task 문자열 | 31 |
| Parquet | 1,200 |
| MP4 | 3,600 |
| 누락/빈 파일 | 0 |
| episode index | 0~1199 연속, 중복 없음 |
| state/action | 모두 14-D |
| FPS/영상 | 15 FPS, 224×224, 카메라 3개 |
| 로컬 크기 | 약 4.5 GB |

---

## 7. GPU 서버로 데이터 전송

### 7.1 목적지 준비

로컬 PC에서 실행한다.

```bash
ssh -J blunex@ai.amrc.kr:21151 \
  -p 31905 root@172.21.121.112 \
  'mkdir -p /mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d'
```

### 7.2 rsync

```bash
rsync -aH --partial --info=progress2 \
  -e "ssh -J blunex@ai.amrc.kr:21151 -p 31905" \
  /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_transport_14d/ \
  root@172.21.121.112:/mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d/
```

MP4는 이미 압축된 데이터이므로 `-z`를 사용하지 않았다. 전송이 중단되면 같은
명령을 다시 실행하여 이어갈 수 있다. source 뒤의 `/`는 디렉터리 자체가 아니라
내용을 목적지 root에 복사한다는 의미다.

내용 checksum 비교가 필요하면 dry-run으로 확인한다.

```bash
rsync -rcn --delete --itemize-changes \
  -e "ssh -J blunex@ai.amrc.kr:21151 -p 31905" \
  /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_transport_14d/ \
  root@172.21.121.112:/mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d/
```

출력이 없으면 source와 destination의 파일 내용이 일치한다. 서버에서
전처리를 시작한 뒤에는 `stats.json`과 보정된 Parquet가 달라지므로 이 명령으로
로컬 원본을 다시 덮어쓰면 안 된다.

---

## 8. 서버 구조 검증과 multi-chunk 수정

서버 접속:

```bash
ssh -J blunex@ai.amrc.kr:21151 \
  -p 31905 root@172.21.121.112

cd /mnt/dev/work/pi05_TO_hybrid
```

검증 명령:

```bash
openpi/.venv/bin/python scripts/validate_dataset.py \
  --dataset /mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d
```

최초 validator는 `chunk-000`만 검색하여 episode 1000~1199의 Parquet 200개와
각 카메라 MP4 200개를 누락으로 잘못 보고했다. 실제 파일은 `chunk-001`에
존재했다. `validate_dataset.py`와 `compute_stats.py`를 다음 glob을 사용하도록
수정했다.

```text
data/chunk-*/episode_*.parquet
videos/chunk-*/observation.images.<camera>/episode_*.mp4
```

수정 후 기대 결과:

```text
parquet                : OK (1200 episodes)
cam cam_high             : OK
cam cam_left_wrist       : OK
cam cam_right_wrist      : OK
```

---

## 9. 두 종류 통계와 전처리

### 9.1 `compute_stats.py`: 원본 데이터 통계

```bash
cd /mnt/dev/work/pi05_TO_hybrid

openpi/.venv/bin/python scripts/compute_stats.py \
  --dataset /mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d \
  2>&1 | tee /mnt/dev/work/pi05_TO_hybrid/logs/rby1_transport_compute_stats.log
```

이 명령은 원본 14-D state/action의 차원별 통계를 계산해 다음에 저장한다.

```text
data/rby1_transport_14d/meta/stats.json
```

계산 항목:

- `min`, `max`
- `mean`, `std`
- `q01`, `q99`

목적은 데이터 분포 분석, 비정상 관절·gripper 값 탐지, LeRobot metadata 완성이다.
실제 실행에서는 1,200/1,200 episode와 609,094 frame을 집계했다.

### 9.2 q01과 q99

- `q01`: 정렬된 값의 하위 1% 지점
- `q99`: 정렬된 값의 상위 1% 지점

양쪽 극단 1%의 outlier가 `min/max` 범위를 과도하게 넓히는 것을 막아 중앙 98%의
일반적 분포를 안정적으로 표현한다. OpenPI의 quantile normalization은 각 state와
action 차원을 대략 일정한 범위로 맞추는 데 이를 사용한다.

### 9.3 `compute_norm_stats.py`: OpenPI 학습용 통계

`compute_stats.py`와 다른 계산이다. OpenPI data config의 실제 변환을 적용한 뒤
학습과 추론에서 사용할 정규화 값을 만든다.

```text
raw observation/action
→ LeRobot feature repack
→ arm joint absolute action을 current state 대비 delta로 변환
→ state/actions의 OpenPI norm stats 계산
→ norm_stats.json
```

학습에서는 이 통계로 입력과 target을 정규화하고, 추론에서는 같은 통계로 모델
action을 역정규화한다. 따라서 checkpoint와 `norm_stats.json`은 항상 함께 보관해야
한다.

---

## 10. Timestamp 문제와 보정

### 10.1 원인

MuJoCo timestep은 0.002초이고 수집 주기는 다음처럼 정수 physics step으로
반올림되었다.

```text
round(1 / (15 × 0.002)) = 33 physics steps
33 × 0.002 = 0.066초
```

하지만 MP4는 nominal 15 FPS CFR이므로 영상 PTS 간격은 다음과 같다.

```text
1 / 15 = 0.0666667초
```

기존 Parquet timestamp는 `0.066, 0.132, ...`였고 MP4 PTS는
`0.0000, 0.0666667, ...`였다. 행과 영상 frame의 물리적 순서는 일치하지만
LeRobot timestamp 검증 및 timestamp 기반 video lookup이 실패했다.

### 10.2 보정 원칙

각 episode의 timestamp만 다음 값으로 교체한다.

```text
timestamp[i] = frame_index[i] / 15
```

다음 항목은 변경하지 않는다.

- state/action
- frame 및 episode index
- task
- MP4 및 frame 순서
- episode 길이

향후 새로 수집하는 데이터는 recorder가 처음부터 이 nominal PTS를 기록하도록
수정했다.

### 10.3 Dry-run과 실제 보정

```bash
openpi/.venv/bin/python scripts/patch_dataset_timestamps.py \
  --dataset /mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d \
  --fps 15 \
  --dry-run

openpi/.venv/bin/python scripts/patch_dataset_timestamps.py \
  --dataset /mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d \
  --fps 15
```

실제 실행 결과:

```text
patched 1200 files, 609094 rows total
timestamp[0] = 0.0000
timestamp[1] - timestamp[0] = 0.066667
```

timestamp만 바뀌므로 이미 계산한 raw `meta/stats.json`은 다시 계산할 필요가 없다.

### 10.4 `.parquet.bak` 중복 로딩 사고

초기 보정 스크립트는 원본을 각 Parquet 옆에 `*.parquet.bak`으로 만들었다.
LeRobot/Hugging Face dataset 생성기가 이 백업도 shard로 인식하여 다음처럼 정확히
두 배를 읽었다.

```text
정상 frame       609,094
백업 포함 split 1,218,188 = 609,094 × 2
```

백업에는 이전 0.066초 timestamp가 남아 있어 normalization이 다시 실패했다.
백업은 삭제하지 않고 다음 외부 경로로 이동했다.

```text
/mnt/dev/work/pi05_TO_hybrid/backups/
  rby1_transport_14d_timestamp_original_20260812/
```

확인 기준:

```text
dataset 내부 *.parquet.bak = 0
외부 backup               = 1,200
정상 *.parquet            = 1,200
```

현재 `patch_dataset_timestamps.py`는 새로 실행할 때 backup을 dataset의 `data/`
밖에 생성하도록 수정되어 있다. 학습 dataset directory 안에는 Parquet, MP4 이외의
backup 파일을 두지 않는 것이 안전하다.

---

## 11. LeRobot 경로와 normalization 실행

### 11.1 기존 repo ID를 위한 심볼릭 링크

현재 `pi05_rby1_lora` config의 repo ID는 다음과 같이 기존 이름으로 되어 있다.

```text
local/rby1_dataset_v1
```

이번 데이터를 별도 LeRobot home에 연결했다.

```bash
cd /mnt/dev/work/pi05_TO_hybrid
mkdir -p lerobot_transport/local

ln -s ../../data/rby1_transport_14d \
  lerobot_transport/local/rby1_dataset_v1
```

이미 존재하면 다시 생성하지 않는다. 확인:

```bash
readlink -f lerobot_transport/local/rby1_dataset_v1
```

기대 경로:

```text
/mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d
```

이 별칭은 config 호환을 위한 것이며 실제 dataset이 이전 block 데이터라는 뜻이
아니다.

### 11.2 새 Hugging Face cache

잘못 생성된 1,218,188-row split cache를 재사용하지 않기 위해 새 cache를
사용한다.

```text
/mnt/dev/work/pi05_TO_hybrid/.cache/hf_datasets_rby1_corrected_20260812
```

### 11.3 normalization 실행

```bash
tmux new-session -d -s rby1_transport_norm \
  "cd /mnt/dev/work/pi05_TO_hybrid/openpi && \
   HF_LEROBOT_HOME=/mnt/dev/work/pi05_TO_hybrid/lerobot_transport \
   HF_DATASETS_CACHE=/mnt/dev/work/pi05_TO_hybrid/.cache/hf_datasets_rby1_corrected_20260812 \
   PYTHONUNBUFFERED=1 \
   .venv/bin/python scripts/compute_norm_stats.py \
     --config-name pi05_rby1_lora \
   2>&1 | tee /mnt/dev/work/pi05_TO_hybrid/logs/rby1_transport_norm.log"
```

검증할 핵심은 다음과 같다.

```text
Generating train split: 609094 examples
```

진행 확인:

```bash
tmux ls
tail -f /mnt/dev/work/pi05_TO_hybrid/logs/rby1_transport_norm.log
```

`tail -f`는 `Ctrl+C`로 종료해도 tmux 내부 작업은 계속된다. 결과 파일:

```text
/mnt/dev/work/pi05_TO_hybrid/openpi/assets/
  pi05_rby1_lora/local/rby1_dataset_v1/norm_stats.json
```

확인:

```bash
test -f \
  /mnt/dev/work/pi05_TO_hybrid/openpi/assets/pi05_rby1_lora/local/rby1_dataset_v1/norm_stats.json \
  && echo "OpenPI norm stats ready"
```

현재 서버 측 자동화는 다음 조건을 모두 통과한 경우에만 학습을 시작한다.

1. split이 정확히 609,094 example
2. normalization process 성공 종료
3. `norm_stats.json` 존재
4. JSON 구문 유효
5. 동일 실험 checkpoint 충돌 없음

---

## 12. π0.5 LoRA 학습

### 12.1 config

`openpi/src/openpi/training/config.py`의 `pi05_rby1_lora`를 사용한다.

| 항목 | 값 |
|---|---|
| Base model | π0.5 base checkpoint |
| Vision/language variant | `gemma_2b_lora` |
| Action expert variant | `gemma_300m_lora` |
| ALOHA hardware 변환 | `adapt_to_pi=False` |
| Joint action | absolute → current-state-relative delta |
| Gripper action | absolute 유지 |
| Batch size | 32 |
| Train steps | 30,000 |
| Warmup | 1,000 |
| Peak learning rate | `5e-5` |
| Final learning rate | `5e-6` |
| LR schedule | cosine decay |
| Save interval | 5,000 step |
| Seed | 42 |
| EMA | 사용하지 않음 |
| WandB | 비활성화 |

H200의 메모리와 저장공간은 충분하지만 첫 재현 학습에서는 검증된 batch 32와
30K 설정을 유지한다. 성능 비교 없이 batch나 learning-rate를 동시에 바꾸면
데이터 변경 효과와 optimizer 변경 효과를 구분하기 어렵다.

### 12.2 실행 명령

현재 서버에 설정된 experiment:

```text
rby1_transport_14d_30k_20260812
```

수동으로 동일하게 실행할 경우:

```bash
tmux new-session -d -s rby1_pi05_lora_30k \
  "cd /mnt/dev/work/pi05_TO_hybrid/openpi && \
   HF_LEROBOT_HOME=/mnt/dev/work/pi05_TO_hybrid/lerobot_transport \
   HF_DATASETS_CACHE=/mnt/dev/work/pi05_TO_hybrid/.cache/hf_datasets_rby1_corrected_20260812 \
   XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
   XLA_PYTHON_CLIENT_PREALLOCATE=false \
   PYTHONUNBUFFERED=1 \
   .venv/bin/python scripts/train.py pi05_rby1_lora \
     --exp-name rby1_transport_14d_30k_20260812 \
     --checkpoint-base-dir /mnt/dev/work/pi05_TO_hybrid/checkpoints \
     --no-wandb-enabled \
     --overwrite \
   2>&1 | tee /mnt/dev/work/pi05_TO_hybrid/logs/rby1_transport_14d_30k_train.log"
```

현재는 norm 완료 후 자동 학습 시작 guard가 이미 구성되어 있으므로 같은 command를
중복 실행하면 안 된다. 먼저 process와 tmux를 확인한다.

```bash
tmux ls
pgrep -af 'compute_norm_stats.py|scripts/train.py'
```

### 12.3 체크포인트와 로그

```text
checkpoint root:
/mnt/dev/work/pi05_TO_hybrid/checkpoints/pi05_rby1_lora/
  rby1_transport_14d_30k_20260812/

training log:
/mnt/dev/work/pi05_TO_hybrid/logs/rby1_transport_14d_30k_train.log
```

config의 5,000-step 저장 간격에 따라 정상 완료 시 대략 다음 step directory가
생성된다.

```text
4999, 9999, 14999, 19999, 24999, 29999
```

진행 확인:

```bash
tmux attach -t rby1_pi05_lora_30k

# 또는 tmux를 attach하지 않고 로그만 확인
tail -f /mnt/dev/work/pi05_TO_hybrid/logs/rby1_transport_14d_30k_train.log
```

새 학습을 시작할 때만 `--overwrite`를 사용한다. 중단된 동일 실험을 이어갈 때는
기존 checkpoint를 덮어쓰지 말고 config가 지원하는 resume 옵션을 사용해야 한다.

---

## 13. 학습 후 보관 및 sim-to-real 연결

학습 산출물을 옮기거나 추론에 사용할 때 다음을 하나의 묶음으로 보관한다.

1. 선택한 checkpoint의 model params
2. checkpoint에 포함되거나 대응하는 `norm_stats.json`
3. 사용한 `pi05_rby1_lora` config revision
4. 세 카메라 key와 전처리 규칙
5. 14-D state/action joint 순서
6. prompt 문자열 규칙

실물 RBY1 적용에서는 모델의 14-D action을 곧바로 SDK에 보내지 않는다. 학습
때 사용한 joint 순서와 단위, absolute/delta 변환, gripper scaling을
`RBY1CommandAdapter`에서 복원한 뒤 SDK 또는 ROS 2 command interface로 전달해야
한다. 또한 joint limit, velocity/acceleration limit, stale observation, self-collision,
workspace constraint를 adapter 앞뒤에서 검사해야 한다.

첫 실물 검증은 다음 순서를 권장한다.

```text
offline dataset replay
→ MuJoCo closed-loop evaluation
→ 실물 shadow mode(명령 전송 없이 action 기록)
→ 낮은 속도·낮은 gain의 single fruit pick
→ pack_only
→ crate lift
→ pack_and_lift
```

장애물 회피는 이번 학습 데이터에 포함되지 않았으므로 별도 safety/planning layer로
평가한다. 구현된 평가 obstacle profile과 wheel 준비 상태는
[`RBY1_OBSTACLE_WHEEL_READINESS_KO.md`](RBY1_OBSTACLE_WHEEL_READINESS_KO.md)를 참고한다.

---

## 14. 문제 해결 체크리스트

### 누락 파일이 정확히 200 episode로 표시됨

validator가 `chunk-000`만 읽는 구버전인지 확인한다. 데이터가 실제로
`chunk-001`에 있으면 재전송 문제가 아니다.

### train split이 1,218,188 examples

정상 609,094의 정확히 두 배이므로 dataset 내부 `*.parquet.bak`을 확인한다.

```bash
find /mnt/dev/work/pi05_TO_hybrid/data/rby1_transport_14d/data \
  -type f -name '*.parquet.bak' | wc -l
```

백업을 dataset 밖으로 이동하고 새로운 `HF_DATASETS_CACHE`로 재실행한다.

### timestamp diff가 약 0.066으로 출력됨

Parquet가 아직 nominal 15 FPS PTS로 보정되지 않았거나 loader가 이전 backup/cache를
읽고 있다. 정상 diff는 float32 오차 범위에서 약 `0.0666667`이다.

### tmux session이 사라짐

정상 완료 또는 오류 종료 둘 다 가능하다. 결과 파일과 로그 끝부분을 함께 본다.

```bash
tmux ls
tail -n 100 /mnt/dev/work/pi05_TO_hybrid/logs/rby1_transport_norm.log
```

### `meta/stats.json`과 `norm_stats.json` 중 하나만 있음

두 파일은 대체 관계가 아니다.

- `meta/stats.json`: raw dataset 분석 및 LeRobot metadata
- `norm_stats.json`: OpenPI 변환 후 학습·추론 정규화

학습 전에는 OpenPI `norm_stats.json`이 반드시 필요하다.

---

## 15. 관련 코드와 문서

| 경로 | 역할 |
|---|---|
| `rby1_manipulation/src/rby1_manipulation/tasks/transport_pack_lift.py` | 단일 transport expert episode |
| `rby1_manipulation/src/rby1_manipulation/data/collect_transport_dataset.py` | 1,200개 균형 schedule 및 resume |
| `rby1_manipulation/src/rby1_manipulation/data/recording.py` | 15 Hz state/action 기록과 병렬 camera rendering |
| `rby1_manipulation/src/rby1_manipulation/data/episode.py` | LeRobot writer와 chunk 관리 |
| `rby1_manipulation/src/rby1_manipulation/planning/transport.py` | fruit/crate waypoint 및 grasp offset |
| `rby1_manipulation/src/rby1_manipulation/simulation/fruit_grid.py` | 16개 fruit grid 구성 |
| `scripts/validate_dataset.py` | multi-chunk 데이터 구조 검증 |
| `scripts/compute_stats.py` | raw state/action `meta/stats.json` |
| `scripts/patch_dataset_timestamps.py` | Parquet timestamp와 video PTS 정렬 |
| `openpi/scripts/compute_norm_stats.py` | OpenPI 변환 후 `norm_stats.json` |
| `openpi/scripts/train.py` | JAX π0.5 LoRA 학습 |
| `openpi/src/openpi/training/config.py` | `pi05_rby1_lora` 설정 |
| `rby1_manipulation/docs/TRANSPORT_DATASET_14D_KO.md` | 수집 schedule과 장면 설계 상세 |
| `docs/RBY1_SIM_TO_REAL_CONTROL_INTERFACES_KO.md` | MuJoCo/SDK/ROS 2 제어 interface와 adapter |
| `docs/RBY1_OBSTACLE_WHEEL_READINESS_KO.md` | 정적·동적 장애물 평가와 wheel readiness gate |
