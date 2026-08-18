# RBY1 장애물 평가와 wheel 구동 준비

## 1. 목적과 범위

현재 학습 중인 `rby1_transport_14d` 정책은 양팔 12축과 양쪽 gripper 2축만
출력한다. wheel 이동과 장애물 회피 데이터는 이 데이터셋에 포함되지 않았다.
따라서 첫 통합 구조는 다음처럼 분리한다.

```text
14-D VLA
  └─ 과일 적재·상자 파지·상자 들기

navigation controller
  └─ wheel의 선속도 v / 각속도 w 및 odometry 폐루프 제어

safety layer
  └─ 장애물 거리 감시 → 감속·정지 → 경로 재계획
```

VLA의 출력을 임의로 17-D 또는 wheel 속도 명령으로 확장하면 학습 때 존재하지
않던 출력 의미를 추론 시 새로 부여하게 된다. 현재 구현은 이를 하지 않으며,
VLA와 navigation을 task state machine으로 연결할 준비만 한다.

장애물 기능은 **평가 전용**이다. `--obstacle-profile`이 `clear`가 아닌 상태에서
`--log-dataset`을 주면 시나리오가 즉시 종료하므로, 기존 1,200-episode 학습
분포에 장애물 장면이 섞이지 않는다.

---

## 2. 정적·동적 장애물 생성 방식

### 2.1 모델 구조

`scenes/transport_obstacles.xml`에 5개의 최대 크기 obstacle slot을 미리 컴파일했다.

| slot | 용도 | primitive |
|---|---|---|
| `static_box_0`, `static_box_1` | 박스·팔레트·고정 구조물 | box |
| `static_column_0` | 기둥·안전봉 | cylinder |
| `dynamic_human_0` | 사람 크기의 이동 장애물 | capsule |
| `dynamic_cart_0` | 카트·운반대 | box |

각 slot은 MuJoCo `mocap` body다. 사용하지 않을 때는 작업장 밖에 주차하고,
profile을 활성화할 때 위치·yaw·크기·색상을 설정한다. 이 방식은 free joint나
actuator를 추가하지 않으므로 기존 모델의 차원을 유지한다.

| 모델 | `nq` | `nv` | `nu` |
|---|---:|---:|---:|
| kinematic transport | 66 | 61 | 29 |
| wheel transport | 66 | 61 | 26 |

설정은 `config/transport_obstacles.json`, 실행 로직은
`simulation/obstacles.py`에 있다.

### 2.2 제공 profile

| profile | 구성 | 검사 목적 |
|---|---|---|
| `clear` | 장애물 없음 | 기존 시나리오 회귀 시험 |
| `static_offset` | 경로 한쪽의 박스 | 여유거리와 우회 경로 시험 |
| `static_blocked` | nominal 경로를 막는 박스 | 정지·재계획 시험 |
| `dynamic_crossing` | 좌우로 왕복하는 사람형 capsule | 동적 정지·재출발 시험 |
| `mixed` | 고정 기둥 + 왕복 카트 | 복합 장면 시험 |

동적 장애물은 `start`, `end`, `speed`, `start_delay`, `phase`, `mode`로 정의한
직선 궤적을 simulation time의 결정론적 함수로 계산한다. 같은 profile과 같은
시작 시간은 항상 같은 위치를 생성하므로 알고리즘 A/B 비교가 가능하다.

현재 구현된 안전 기능은 다음과 같다.

- 로봇·상자·과일과 obstacle collision contact 기록
- 원형 base footprint를 이용한 보수적 2-D 최소거리 기록
- kinematic/wheel base 주행 중 5 cm 미만 또는 collision 발생 시 emergency stop
- 충돌 step 수, 충돌 geom pair, 최소 clearance 출력

아직 구현되지 않은 기능은 **우회 경로 생성과 재출발 판단**이다. 따라서
`static_blocked`에서 contact 전에 안전하게 멈추는 것은 safety 동작의 성공이고,
운반 task의 목적지 도달 결과는 실패가 맞다. 안전 정지 후 place 단계도 실행하지 않는다.

### 2.3 실행 방법

workspace에서 editable install을 한 뒤 실행한다.

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
python -m pip install -e src/rby1_manipulation

# profile만 빠르게 확인
python -m rby1_manipulation.tools.preview_transport_obstacles \
  --profile static_offset
python -m rby1_manipulation.tools.preview_transport_obstacles \
  --profile dynamic_crossing

# GUI 없이 5초간 결정론적 동작 확인
python -m rby1_manipulation.tools.preview_transport_obstacles \
  --profile dynamic_crossing --duration 5 --headless

# 전체 manipulation 시나리오에 평가 장애물 적용
python -m rby1_manipulation.tasks.transport_load_and_carry \
  --headless --object apple --obstacle-profile static_blocked
```

custom JSON은 `--obstacle-config /path/to/config.json`으로 주입한다. JSON 크기는
XML에 컴파일한 각 slot의 최대 크기를 넘을 수 없으며, validator가 시작 시 이를
검사한다.

---

## 3. wheel 가동 준비

### 3.1 현재 만들어진 인터페이스

`model_transport_wheels.xml`은 base 위치 actuator 대신 좌우 wheel velocity
actuator를 사용한다. `control/mobile_base.py`에는 다음이 준비되어 있다.

- `(v, w)`를 좌·우 wheel rad/s로 변환하는 differential-drive kinematics
- 실제 joint axis에 맞춘 부호: 전진은 양쪽 음수, `L+ / R-`는 +yaw
- wheel actuator 이름·joint 연결·velocity servo·control range 사전점검
- odometry 역할을 하는 MuJoCo base pose 기반 폐루프 제어
- 선속도·각속도 가속도 제한
- 목표 도달, timeout, stall, safety stop을 구분하는 결과 객체
- 장애물 safety callback

사전점검은 다음과 같이 실행한다.

```bash
# 정적 인터페이스 검사: 현재 PASS
python -m rby1_manipulation.tools.check_wheel_drive

# 실제 회전 90도 + 직진 0.5 m 검사: 현재 FAIL이 알려진 상태
python -m rby1_manipulation.tools.check_wheel_drive --motion-test
echo $?
```

### 3.2 중요한 현재 판정

| gate | 현재 상태 | 의미 |
|---|---|---|
| XML/actuator preflight | PASS | wheel 명령을 보낼 인터페이스는 올바름 |
| 직진 단독 실험 | 과거 PASS | 양 wheel의 전진 구름은 확인됨 |
| 90도 회전 + 0.5 m 연속 시험 | FAIL | route controller로 사용할 준비는 아직 안 됨 |
| 장애물 emergency stop | 구현됨 | 회피 planner는 아직 없음 |
| 전체 상자 운반 wheel 성공 | 미통과 | 실물 적용 gate를 통과하지 못함 |

제자리 회전 실패의 주원인은 planar base에 z/suspension 자유도가 없어서 wheel
접촉 normal force가 비현실적으로 커지는 현재 MuJoCo 물리 구조다. controller gain만
높이는 방식은 근본 해결이 아니며 수치 발산 위험이 있다.

따라서 다음 단계는 기존 14-D 데이터 모델을 변경하지 않고 **별도의 wheel 평가
모델**에 base vertical/suspension 자유도를 추가하는 것이다. 이 모델에서 아래 gate를
순서대로 통과해야 한다.

1. 정지 상태에서 base 높이와 wheel normal force가 실제 중량 범위로 수렴
2. 전진·후진 0.5 m 오차 2 cm 이하
3. 좌·우 90도 회전 오차 2도 이하
4. 사각 경로 종료 pose 오차 5 cm / 3도 이하
5. crate를 든 상태에서 같은 경로 통과
6. `static_blocked` emergency stop, `static_offset` 우회, `dynamic_crossing` 정지·재출발

### 3.3 실물 RBY1 연결 원칙

MuJoCo의 absolute base pose 명령을 실물에 그대로 보내지 않는다. 실물 adapter는
목표 pose를 localization/odometry와 비교해 bounded `(v, w)`로 만들고, 이를 RBY1
SDK wheel command 또는 ROS 2 mobile-base velocity interface로 전달해야 한다.

필수 안전 조건은 다음과 같다.

- 속도·가속도·jerk limit
- command watchdog과 deadman stop
- odometry timeout 및 localization jump 검사
- 최소 장애물 거리에서 독립적인 zero-velocity override
- VLA/arm failure 시 base 즉시 정지
- sim joint/actuator 이름과 real SDK/ROS joint 이름의 명시적 adapter mapping

첫 실물 시험은 바퀴를 띄운 상태의 부호 검사, 저속 직진, 저속 회전, 빈 로봇 경로,
빈 상자 운반, 적재 상자 운반 순으로 진행한다.

---

## 4. 추가 개선 항목

### P0: wheel 물리와 안전 gate

- suspension이 있는 전용 wheel 평가 모델
- clear 경로 회귀 시험을 CI와 별도 slow test로 추가
- emergency stop 후 wheel command가 0인지 검증
- 실제 RBY1 최대 속도·가속도와 동일한 limit 적용

### P1: 회피 알고리즘

- 첫 단계: MuJoCo ground-truth obstacle pose로 local planner 검증
- 정적 장애물: inflation map + A*/Hybrid A* 또는 waypoint detour
- 동적 장애물: time-to-collision 기반 감속·정지 후 재출발
- 이후 단계: depth/LiDAR/robot perception 결과로 ground truth를 교체
- 평가 지표: 성공률, collision 수, 최소거리, 정지거리, 경로 길이, 소요시간,
  불필요한 정지 횟수

학습 없는 회피가 목적이므로 obstacle observation을 14-D VLA 입력에 추가하기보다
navigation safety/planning layer가 소유한다.

### P2: sim-to-real와 통합 안정성

- wheel friction, payload mass, command delay, odometry noise domain randomization
- arm 작업과 navigation 사이의 명시적 상태: `PACKED → LIFTED → NAVIGATING → PLACING`
- 이동 전 crate grasp·과일 containment 확인, 실패 시 navigation 금지
- base 이동 중 arm hold controller와 crate slip detector
- 학습 checkpoint별 고정 seed 평가 리포트 자동 생성

---

## 5. 구현 파일

| 파일 | 역할 |
|---|---|
| `rby1_description/.../scenes/transport_obstacles.xml` | 고정 크기의 obstacle slot |
| `rby1_manipulation/.../config/transport_obstacles.json` | 평가 profile |
| `rby1_manipulation/.../simulation/obstacles.py` | 배치·동작·거리·충돌 감시 |
| `rby1_manipulation/.../control/mobile_base.py` | wheel preflight와 폐루프 controller |
| `rby1_manipulation/.../tools/preview_transport_obstacles.py` | profile preview |
| `rby1_manipulation/.../tools/check_wheel_drive.py` | wheel readiness gate |
| `rby1_manipulation/.../tasks/transport_load_and_carry.py` | 평가 profile 통합 |
| `rby1_manipulation/tests/test_obstacle_wheel_readiness.py` | 차원·궤적·명령 contract |
