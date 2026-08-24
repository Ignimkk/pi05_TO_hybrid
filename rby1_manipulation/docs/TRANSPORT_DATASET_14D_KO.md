# 고정 베이스 과일 적재·상자 들기 14-D 데이터셋

수집 완료 결과, GPU 서버 전송, multi-chunk/timestamp 전처리와 π0.5 LoRA 학습까지
이어지는 전체 절차는
[`docs/RBY1_TRANSPORT_14D_DATA_PIPELINE_KO.md`](../../docs/RBY1_TRANSPORT_14D_DATA_PIPELINE_KO.md)를
참고한다. 이 문서는 수집 schedule과 장면 설계에 집중한다.

## 수집 범위

이번 데이터셋은 `transport_load_and_carry`의 과일 pick과 양팔 crate grasp를
재사용하지만 베이스 이동, wheel 제어, 선반 배치를 수행하지 않는다. 정적·동적
장애물도 포함하지 않는다.

관측과 action은 기존 block 데이터와 같은 `rby1_14`이다.

```text
[left arm 0..5, left gripper,
 right arm 0..5, right gripper]
```

카메라는 기존과 동일한 `cam_high`, `cam_left_wrist`, `cam_right_wrist` 224×224를
사용한다.

GPU가 없는 OSMesa 환경에서는 policy camera의 scene reflection만 비활성화한다.
카메라 시점, RGB 해상도, lighting, shadow, 15 FPS는 유지하며, 이 설정은 3-camera
software rendering benchmark에서 약 2.2배 빠르다.

`transport_pack_lift`는 시뮬레이션 도중 state/action과 MuJoCo `qpos`를 15 FPS로
먼저 기록하고, 성공 판정 후 세 policy camera를 별도 프로세스에서 병렬
렌더링한다. 관측 시점과 데이터 포맷은 기존과 같지만 세 카메라의 순차 렌더링
대기 시간을 줄인다. 저장하지 않는 실패 에피소드는 policy camera 렌더링을
수행하지 않는다.

## 1,200 episode 균형

아래 표는 최초 데이터셋 `rby1_transport_14d`의 구성이다.

| 지시문 계열 | 수량 | 내부 구성 |
|---|---:|---|
| `pack_only` | 400 | 적재 수 1·2·3·4개를 각 100개 |
| `lift_only` | 400 | 사전 적재 수 0·1·2·3·4개를 각 80개 |
| `pack_and_lift` | 400 | 적재 수 1·2·3·4개를 각 100개 |

과일 노출 수는 다음과 같다.

| 계열 | apple | banana | orange | pear |
|---|---:|---:|---:|---:|
| `pack_only` | 250 | 250 | 250 | 250 |
| `lift_only` 사전 적재 | 200 | 200 | 200 | 200 |
| `pack_and_lift` | 250 | 250 | 250 | 250 |

과일 실행 순서는 24개 순열을 순환한다. prompt는 선택된 과일 집합을 나타내며,
동일한 prompt 안에서도 expert의 적재 순서는 균형 있게 달라진다.

## 재수집 v2: lift-only 절반 축소

확장된 테이블과 강화된 팔 교대 복귀 로직으로 다시 수집하는
`rby1_transport_14d_v2`는 최초 데이터셋의
`transport_episode_manifest.jsonl`을 episode 사양의 기준으로 사용한다. 기존
`pack_only`와 `pack_and_lift` 800개는 그대로 유지하고, `lift_only`만 400개에서
200개로 줄여 총 1,000개를 수집한다.

| 지시문 계열 | 수량 | 구성 |
|---|---:|---|
| `pack_only` | 400 | 기존 manifest 사양 전체 유지 |
| `lift_only` | 200 | 사전 적재 0·1·2·3·4개를 각 40개 |
| `pack_and_lift` | 400 | 기존 manifest 사양 전체 유지 |

축소된 `lift_only`도 임의 추출에 맡기지 않고 균형 제약으로 선택한다. 각 과일은
상자 안에 100회씩 등장한다. 전체 1,000개에서 각 과일은 table/crate slot
0·1·2·3에 각각 250회 등장하고, 16개 grid layout은 각 62~63회 사용한다.
선택은 `--selection-seed 20260820`으로 재현할 수 있으며 기준 manifest의 SHA-256도
수집 plan에 저장한다.

## Fruit grid

설정은 `config/grids/fruit_grid.json`에 있다. 좌우에 각각 4개 후보점이 있고,
같은 쪽 과일 두 개가 최소 75mm 이상 떨어지는 pair만 사용한다. 좌 pair 4개와
우 pair 4개의 조합으로 총 16개 layout을 만든다.

- 각 layout: 75 episode
- 각 과일: table slot 0·1·2·3에 각각 300회
- 각 장면: 과일 2개는 왼팔 영역, 2개는 오른팔 영역
- crate 내부: 충돌을 줄이는 4개 local drop slot 사용

미리 적재된 `lift_only` 장면도 동일한 grid를 사용한다. 상자 안에 들어간 과일을
제외한 나머지 과일은 table grid에 distractor로 남는다.

미리보기:

```bash
python -m rby1_manipulation.tools.preview_fruit_grid --layout-index 0

python -m rby1_manipulation.tools.preview_fruit_grid \
  --layout-index 15 \
  --slot-order pear orange banana apple \
  --preloaded apple banana \
  --headless
```

## 단일 episode 실행

동작 시간은 기본 `1.25x`이다. 이동 경로는 바뀌지 않고 waypoint ramp, waypoint 사이
대기, gripper 안정화 시간이 기존 대비 `1/1.25`로 줄어든다. 기존 속도와 비교하려면
`--speed-scale 1.0`을 지정한다.

작은 물체의 마지막 grasp 접근 후 손가락을 더 빨리 닫고 싶으면
`--object-pre-close-hold 0.25`를 지정한다. 옵션을 생략하면 검증된 기본 대기를
유지한다. 0.20초 이하에서는 과일 grasp는 성공해도 낙하 상태 변화로 후속 crate
grasp가 불안정해진 사례가 있어 전체 수집 기본값에는 적용하지 않는다.

```bash
# 과일만 적재
python -m rby1_manipulation.tasks.transport_pack_lift \
  --task pack_only --objects apple banana --headless

# 과일이 미리 들어 있는 상자만 들기
python -m rby1_manipulation.tasks.transport_pack_lift \
  --task lift_only --preloaded apple orange pear --headless

# 과일을 적재한 뒤 상자 들기
python -m rby1_manipulation.tasks.transport_pack_lift \
  --task pack_and_lift --objects banana apple --headless
```

더 빠른 값도 지정할 수 있지만, `1.5x` 검증에서는 banana가 운반 중 이탈했다.
따라서 `1.25x`를 수집 기본값으로 사용하고 그보다 빠른 값은 한계 시험에만 사용한다.

```bash
python -m rby1_manipulation.tasks.transport_pack_lift \
  --task pack_and_lift --objects apple banana \
  --speed-scale 1.5 --headless
```

모든 모드에서 base target을 생성하지 않으며 최종 base drift가 0.01m/rad 미만인지
검사한다.

여러 과일을 적재할 때 다음 과일의 담당 팔이 바뀌면, 이전 팔은 release와 수직
retract 후 초기 arm joint 자세로 복귀한다. 예를 들어 `left → right` 순서에서는
left arm이 crate 위 공유 작업공간에서 빠진 다음 right arm이 시작한다. 같은 팔이
연속으로 작업하면 중간 복귀를 생략한다. 이 동작은 에피소드 종료 후 reset이 아니라
한 에피소드 내부의 팔 교대 안전 동작이며 state/action에 함께 기록된다.
복귀 명령을 보낸 직후 다음 팔을 시작하지 않고, 실제 측정 joint가 초기 자세에서
최대 0.02rad 이내로 수렴할 때까지 기다린다. 1초 안에 수렴하지 않으면 해당 시도는
저장하지 않는다. `pack_and_lift`는 상자를 들기 전에도 양팔에 같은 검사를 적용한다.
또한 좌우 arm subtree 사이 contact를 100Hz로 검사하며, 양팔 접촉이 한 번이라도
검출된 episode는 성공 데이터로 저장하지 않는다.

## 전체 수집

먼저 schedule을 확인한다.

```bash
python -m rby1_manipulation.data.collect_transport_dataset \
  --output-dir /data/rby1_transport_14d \
  --dry-run --dry-run-limit 12
```

전체 1,200개 성공 episode 수집:

```bash
python -m rby1_manipulation.data.collect_transport_dataset \
  --output-dir /data/rby1_transport_14d \
  --speed-scale 1.25
```

기존 manifest를 기준으로 v2 1,000개를 새 디렉터리에 수집:

```bash
python -m rby1_manipulation.data.collect_transport_dataset \
  --output-dir /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_transport_14d_v2 \
  --reference-manifest /home/mk/dev_ws/vla/pi0_TO_ws/datasets/rby1_transport_14d/transport_episode_manifest.jsonl \
  --lift-only-episodes 200 \
  --selection-seed 20260820 \
  --speed-scale 1.25
```

v2는 대칭으로 20cm 확장한 0.50×1.00m 테이블을 사용한다. wheel, 선반 운반,
정적·동적 장애물은 계속 비활성화되어 있다. 기존 `rby1_transport_14d`는 수정하지
않는다.

기본적으로 실패한 plan은 장면 randomization seed를 바꾸어 최대 5회 시도한다.
성공한 plan만 LeRobot dataset에 저장된다. 중단 후 같은 명령을 다시 실행하면
`transport_collection_stats.json`을 읽어 성공한 plan을 건너뛴다. 더 많은 retry가
필요하면 `--max-attempts 10`처럼 기존 값보다 크게 지정한다.

출력 root의 sidecar:

- `transport_collection_plan.json`: 전체 schedule과 분포
- `transport_collection_stats.json`: plan별 시도·성공·마지막 오류
- `transport_episode_manifest.jsonl`: 저장 episode와 grid/과일/seed 대응

## Domain randomization

수집 기본값은 다음 범위만 사용한다.

- crate xy jitter: ±5mm
- crate yaw: 0rad 고정
- crate mass: 0.65~1.0kg
- crate friction multiplier: 0.9~1.2

crate yaw는 반복 fruit drop 후 handle grasp 성공률을 낮추므로 이번 expert dataset에서는
변경하지 않는다. scene 다양성은 16개 grid, 24개 과일 순열, 적재 수와 사전 적재
조합에서 확보한다.

장애물은 이 데이터셋에 포함하지 않는다. 향후 collision-avoidance 알고리즘 평가는
별도 evaluation scene/config로 추가하여 학습 데이터와 혼합하지 않는다.

banana는 낮고 휘어진 collision mesh 때문에 일반 mesh 높이식이 계산한 grasp offset이
약 13.2mm에 불과했다. finger-table 충돌 방지를 위해 banana의 최소 grasp offset을
22mm로 적용한다. 모든 과일 pick 구간에서 finger-table 접촉을 100Hz로 검사하며,
접촉이 한 번이라도 검출된 episode는 저장하지 않는다.

초기 수집에서 실패 plan 91개 중 88개는 pear가 close 직후 접촉 판정을 통과하고도
첫 lift에서 빠지는 문제였다. mesh 높이식의 25.9mm offset이 좁은 목 부분을 잡았기
때문이며, 넓은 몸통을 잡도록 pear offset을 15mm로 지정한다. 좌우 8개 grid 위치와
실제 실패 plan을 재현하여 table 접촉 없이 pear 적재가 성공하는 것을 확인했다.
