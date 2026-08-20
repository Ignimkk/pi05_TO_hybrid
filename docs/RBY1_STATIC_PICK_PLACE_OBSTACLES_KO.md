# RBY1 정적 pick-and-place 장애물 평가

## 1. 목적과 범위

이 구현의 주 대상은 현재 학습 중인 `rby1_transport_14d` 정책이다.

- 과업: 사과·바나나·오렌지·배를 상자에 넣거나 상자를 들어 올리기
- 정책 입출력: 양팔 12개 관절 + 양쪽 gripper 2개인 14-D
- 장애물: 작업대에 고정된 tabletop post 또는 낮은 divider
- 실행기: `rby1_bringup/pi05_infer.py`, `pi05_ex_infer.py`
- 보조 지원: 이전 `rby1_dataset_v1` 블록 pick-and-place 정책
- 이번 범위에서 제외: wheel 주행 장애물, 동적 장애물

wheel 장애물은 wheel 물리와 이동 제어를 다시 검증한 뒤 별도의 navigation scene으로
구현한다. 여기서 다루는 장애물은 모두 작업대에 고정되며 시간이 지나도 움직이지 않는다.

중요한 제한이 있다. 현재 1,200개 과일 데이터셋에는 장애물 회피 시연이 없으므로, 정적
장애물을 영상에 추가하는 것만으로 VLA가 회피한다고 보장할 수 없다. 이번 구현은 다음 두
가지를 가능하게 한다.

1. 학습하지 않은 장면에서 정책의 zero-shot 회피 여부를 실제 inference loop로 평가한다.
2. 정책이 장애물로 접근하면 MuJoCo geometry 기반 safety layer가 독립적으로 정지한다.

safety layer는 경로를 새로 만드는 planner가 아니다. `obstacle_stop`은 로봇 충돌 방지에는
성공했지만 조작 과업은 실패한 결과다.

---

## 2. scene과 14-D interface

원본 학습 scene은 그대로 보존하고, 장애물 추론에서만 별도 모델을 선택한다.

| 정책/scene | 원본 모델 | 정적 장애물 모델 |
|---|---|---|
| 과일–상자 14-D | `model_transport.xml` | `model_transport_pick_place_obstacles.xml` |
| 블록 14-D | `model.xml` | `model_pick_place_obstacles.xml` |

과일 모델은 장애물을 추가한 뒤에도 `(nq, nv, nu)=(66, 61, 29)`와 actuator 순서가 원본과
같다. `nu=29` 중 기존 transport 모델에 있던 base 위치 actuator 3개는 초기값으로 유지하며,
정책은 학습 때와 동일하게 양팔·gripper 14-D만 관측하고 명령한다. 장애물은 mocap body로
구현했기 때문에 qpos, state, action 차원을 늘리지 않는다.

```text
rby1_transport_14d action (14)
  → 기존 RBY1 14-D action adapter
  → arm/gripper actuator target

static obstacle profile
  → parked mocap slot을 고정 좌표로 이동
  → 세 policy camera에 자동으로 렌더링
  → 14-D observation/action 형식에는 변화 없음
```

---

## 3. 장애물 모델

장애물 MJCF는
`rby1_description/models/rby1a/mujoco/scenes/pick_place_obstacles.xml`에 있다.

| 종류 | 크기와 외형 | 평가 목적 |
|---|---|---|
| tabletop post | post 반경 27 mm, foot 반경 45 mm, 전체 높이 287 mm | 과일 옆의 좁은 접근에서 팔·gripper가 여유를 유지하는지 확인 |
| low divider | 폭 208 mm, 높이 80 mm, 깊이 36 mm panel과 깊이 80 mm foot | 낮은 장애물 위로 물체를 충분히 들어 운반하는지 확인 |

보이는 geom과 collision geom은 분리했다. collision geom은 post의 cap/foot 및 divider의
frame/foot까지 포함하며 viewer group 3에 숨겨진다. policy camera에는 완성된 prop 외형만
표시된다. mocap body는 자유 낙하하지 않으므로 정적 장애물로 고정된다.

---

## 4. 과일용 profile

설정 파일은
`rby1_manipulation/src/rby1_manipulation/config/pick_place_obstacles.json`이다.

| profile | 위치/용도 | 기본 keyframe에서 가까운 과일 |
|---|---|---|
| `clear` | 장애물 없는 회귀 기준 | 전체 |
| `fruit_right_bollard` | 오른쪽(−Y) pickup corridor 옆 post | 사과, 오렌지 |
| `fruit_left_bollard` | 왼쪽(+Y) pickup corridor 옆 post | 바나나, 배 |
| `fruit_right_divider` | 오른쪽 pickup corridor 옆 low divider | 사과, 오렌지 |
| `fruit_left_divider` | 왼쪽 pickup corridor 옆 low divider | 바나나, 배 |
| `fruit_dual_bollards` | 양쪽에 post 한 개씩 | 여러 과일/양손 과업 |

초기 keyframe에서 검증한 값은 다음과 같다.

| profile 종류 | 초기 robot clearance | 초기 movable-object clearance | 초기 contact |
|---|---:|---:|---:|
| 한쪽 post | 약 0.066 m | 약 0.017 m | 없음 |
| 한쪽 divider | 약 0.188–0.190 m | 약 0.075 m | 없음 |
| 양쪽 post | 약 0.066 m | 약 0.017 m | 없음 |

16개 grid × 과일 순열 4개 × 장애물 profile 5개인 320개 reset 조합도 별도로 검사했다.
그 조합들에서 robot 최소 여유는 약 0.070 m, movable-object 최소 여유는 0.010 m였고
초기 overlap은 없었다.

블록용 `right_bollard`, `left_bollard`, `right_divider`, `left_divider`,
`dual_bollards`도 유지한다. 과일 프로파일과 블록 프로파일을 잘못 조합하면 CLI가 실행 전에
거부한다.

---

## 5. 추론 전에 policy camera 확인

개발 환경을 활성화한 뒤 실행한다.

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
python -m pip install -e src/rby1_manipulation
```

MuJoCo viewer에서 과일 scene을 확인한다.

```bash
python -m rby1_manipulation.tools.preview_pick_place_obstacles \
  --scene fruit \
  --profile fruit_right_bollard

python -m rby1_manipulation.tools.preview_pick_place_obstacles \
  --scene fruit \
  --profile fruit_right_divider
```

정책에 실제 전달되는 세 장의 224×224 영상을 저장한다.

```bash
python -m rby1_manipulation.tools.preview_pick_place_obstacles \
  --scene fruit \
  --profile fruit_right_bollard \
  --headless \
  --output-dir /tmp/rby1_fruit_obstacle_preview
```

추론에 사용할 grid/preload 상태까지 똑같이 preview할 수 있다.

```bash
python -m rby1_manipulation.tools.preview_pick_place_obstacles \
  --scene fruit \
  --profile fruit_right_bollard \
  --fruit-layout-index 0 \
  --fruit-slot-order banana pear apple orange \
  --headless \
  --output-dir /tmp/rby1_fruit_obstacle_preview
```

다음 파일을 반드시 확인한다.

```text
fruit_right_bollard_cam_high.png
fruit_right_bollard_cam_left_wrist.png
fruit_right_bollard_cam_right_wrist.png
fruit_right_bollard_policy_cameras.png
```

장애물이 보이는지만 확인하지 말고 target fruit, crate, gripper를 지나치게 가리는지도 확인한다.
preview는 시작 시 robot/object clearance와 contact pair도 출력한다.

---

## 6. 현재 학습한 정책으로 실제 추론

### 6.1 원격 policy server 사용

GPU 서버에서 policy server가 실행 중이고 로컬 MuJoCo client가 `HOST:PORT`에 연결하는
구성이다. 서버는 장애물이 포함된 세 camera image와 기존 14-D state를 받는다. 장애물
scene 선택과 100 Hz safety 검사는 로컬 client가 담당하므로 서버의 MuJoCo 모델을 바꿀
필요가 없다.

먼저 동일 prompt를 장애물 없이 실행해 checkpoint 자체의 성공 여부를 확인한다.

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws

python src/rby1_bringup/pi05_infer.py \
  --model rby1_transport_14d \
  --remote HOST:PORT \
  --prompt "put the apple in the crate" \
  --obstacle-profile clear \
  --max-steps 900 \
  --speed 1.0
```

데이터 수집 때 사용한 16개 fruit-grid 중 하나와 과일 순열까지 고정하려면 다음 옵션을
추가한다. `clear`와 장애물 조건에서 이 세 옵션을 똑같이 사용해야 공정한 비교가 된다.

```bash
  --fruit-layout-index 0 \
  --fruit-slot-order apple banana orange pear
```

그 다음 같은 prompt와 초기 상태에서 오른쪽 정적 post 조건을 실행한다.

```bash
python src/rby1_bringup/pi05_infer.py \
  --model rby1_transport_14d \
  --remote HOST:PORT \
  --prompt "put the apple in the crate" \
  --fruit-layout-index 0 \
  --fruit-slot-order banana pear apple orange \
  --obstacle-profile fruit_right_bollard \
  --obstacle-stop-distance 0.02 \
  --max-steps 900 \
  --speed 1.0 \
  --view front \
  --record data/obstacle_eval/apple_right_bollard.mp4 \
  --record-inputs data/obstacle_eval/apple_right_bollard_inputs \
  --trajectory-out data/obstacle_eval/apple_right_bollard.npz
```

왼쪽 과일에는 왼쪽 profile을 사용한다.

```bash
python src/rby1_bringup/pi05_infer.py \
  --model rby1_transport_14d \
  --remote HOST:PORT \
  --prompt "put the banana in the crate" \
  --obstacle-profile fruit_left_divider \
  --obstacle-stop-distance 0.02 \
  --max-steps 900 \
  --speed 1.0
```

`lift the crate`처럼 학습 데이터에서 과일이 미리 들어 있던 조건은 preload를 명시한다.

```bash
python src/rby1_bringup/pi05_infer.py \
  --model rby1_transport_14d \
  --remote HOST:PORT \
  --prompt "lift the crate" \
  --fruit-layout-index 0 \
  --fruit-slot-order apple banana orange pear \
  --fruit-preloaded apple banana \
  --obstacle-profile fruit_dual_bollards \
  --obstacle-stop-distance 0.02 \
  --max-steps 900 \
  --speed 1.0
```

`--fruit-preloaded` 뒤에는 상자 안에 미리 둘 과일을 0–4개 지정한다. preload와 grid
metadata는 `--trajectory-out`으로 만든 NPZ에도 함께 저장된다.

네 과일과 양쪽 장애물을 평가하는 예시는 다음과 같다. 과업 길이가 길기 때문에 step 제한을
충분히 크게 둔다.

```bash
python src/rby1_bringup/pi05_infer.py \
  --model rby1_transport_14d \
  --remote HOST:PORT \
  --prompt "put all four fruits in the crate" \
  --obstacle-profile fruit_dual_bollards \
  --obstacle-stop-distance 0.02 \
  --max-steps 1400 \
  --speed 1.0
```

prompt는 데이터셋의 task 문자열과 정확히 맞추는 것을 권장한다. 예를 들면
`put the apple in the crate`, `put the apple in the crate and lift the crate`,
`put all four fruits in the crate`, `lift the crate`이다.

### 6.2 로컬 checkpoint 사용

원격 서버 대신 checkpoint를 로컬에서 로드한다.

```bash
python src/rby1_bringup/pi05_infer.py \
  --model rby1_transport_14d \
  --checkpoint /path/to/checkpoint/29999 \
  --prompt "put the orange in the crate" \
  --obstacle-profile fruit_right_divider \
  --obstacle-stop-distance 0.02 \
  --max-steps 900 \
  --speed 1.0
```

`--remote`를 사용하면 `--checkpoint`는 로컬 client에 필요하지 않다.

---

## 7. inference safety 동작

policy action은 기존처럼 15 Hz로 실행한다. 각 action chunk를 실행하는 동안 양팔과 양쪽
gripper의 obstacle clearance는 physics loop에서 5 simulation step마다, 즉 100 Hz로
확인한다.

```text
policy action 수신
  → arm/gripper actuator target 적용
  → physics step
  → robot-obstacle contact와 최소거리 검사 (100 Hz)
  → distance <= 0.02 m 또는 contact이면 현재 qpos로 모든 position actuator hold
  → 현재 action chunk 및 rollout 중단
  → obstacle_stop과 safety summary 기록
```

추론 시작 전에도 다음을 검사하며 조건을 만족하지 않으면 실행 자체를 거부한다.

- robot clearance가 `--obstacle-stop-distance`보다 커야 함
- obstacle과 과일/상자가 겹치지 않아야 함
- profile의 `scene`이 선택한 model과 같아야 함

`--obstacle-stop-distance`는 MuJoCo collision geometry 기준이다. 기본 2 cm에서 시작하고,
영상과 `min_robot_clearance_m`, `robot_collision`, `payload_collision`, `contact_pairs`를 함께
확인한다.

---

## 8. 권장 평가 순서

각 prompt/profile 조합을 바로 한 번만 비교하지 말고 다음 순서를 사용한다.

1. `clear` 조건을 여러 seed에서 실행해 정책의 기본 성공률을 측정한다.
2. 동일 prompt와 초기 배치에 한쪽 post를 추가한다.
3. 같은 쪽 low divider로 바꿔 장애물 형상 영향을 비교한다.
4. 여러 과일 과업에서 `fruit_dual_bollards`를 실행한다.
5. task success, `obstacle_stop`, 실제 contact, 최소 clearance, 완료 시간을 별도로 집계한다.

학습하지 않은 회피를 평가하려면 clear 조건 실패와 장애물 때문에 발생한 실패를 섞지 않는
것이 중요하다. 영상은 third-person 영상뿐 아니라 세 policy-input 영상도 함께 저장한다.

---

## 9. custom 정적 장애물 위치

새 profile은 원본 JSON을 복사한 뒤 `scene`, `slot`, `position`, `yaw`를 바꿔 만든다.

```json
{
  "scene": "fruit",
  "description": "custom right-side tabletop post",
  "obstacles": [
    {"slot": "bollard_0", "position": [0.62, -0.25, 0.82], "yaw": 0.0}
  ]
}
```

실행 시 별도 설정을 선택한다.

```bash
python src/rby1_bringup/pi05_infer.py \
  --model rby1_transport_14d \
  --remote HOST:PORT \
  --prompt "put the apple in the crate" \
  --obstacle-config /path/to/custom_pick_place_obstacles.json \
  --obstacle-profile my_profile
```

좌표를 직접 바꾼 뒤 곧바로 inference를 실행하지 말고, 반드시 preview에서 초기 contact가
없고 robot clearance가 2 cm보다 큰지 확인한다. 과일 grid나 preload 상태가 달라진다면 해당
reset 상태에 대해서도 같은 검사를 다시 해야 한다.

---

## 10. 실물 로봇으로 옮길 때

MuJoCo manager가 사용하는 `mj_geomDistance`는 simulation ground truth이므로 실물 RBY1에서
그대로 사용할 수 없다. 실물에서는 같은 safety interface의 거리 입력을 다음 중 하나로
교체해야 한다.

- depth camera point cloud와 robot self-filter
- MoveIt PlanningScene의 collision object와 distance/contact 검사
- 작업대에 고정된 장애물이라면 사전에 측정한 pose를 robot base frame으로 변환한 collision map

VLA의 14-D 명령은 기존 `RBY1CommandAdapter`를 통해 실물 SDK/ROS 2 명령으로 변환하고,
safety layer는 그 명령을 보내기 전에 감시·중단해야 한다. 현재 구현은 MuJoCo 추론 평가용이며,
실물 자동 회피나 실물 emergency stop을 대신하지 않는다.
