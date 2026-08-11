# RBY1 MuJoCo · SDK · ROS 2 제어 인터페이스와 sim-to-real 설계

이 문서는 이 저장소에서 개발한 RBY1 VLA 알고리즘을 MuJoCo에서 검증한 뒤
실물 RBY1에 적용하기 위해 다음 세 경로를 비교하고, 현재 코드의 구조와 향후
`RBY1CommandAdapter` 설계를 정리한다.

1. 현재 저장소의 MuJoCo 직접 제어
2. [`rby1-sdk`](https://github.com/RainbowRobotics/rby1-sdk)를 통한 실물 제어
3. [`rby1-ros2`](https://github.com/RainbowRobotics/rby1-ros2)를 통한 실물 제어

외부 인터페이스 내용은 **2026-08-10** 기준 공식 main 문서와 소스를 확인한
결과이다. RBY1 ROS 2 driver는 현재 beta이므로 설치한 commit에서 실제 action,
service, topic 이름을 다시 확인해야 한다.

---

## 1. 결론

현재 저장소에는 이름과 책임이 분리된 `RBY1CommandAdapter` 클래스가 **없다**.
그 역할의 일부가 다음 함수와 실행 파일에 분산되어 있다.

| 현재 코드 | 수행 중인 adapter 역할 | 결합된 backend |
|---|---|---|
| `pi05_infer.py::build_obs()` | MuJoCo 상태와 영상을 정책 관측 형식으로 변환 | MuJoCo |
| `pi05_infer.py::apply_action()` | 정책 action 해석, gripper 정규화 해제, actuator 매핑 | MuJoCo `data.ctrl` |
| `pi05_infer.py::rby1_state()` | MuJoCo `qpos`를 14-D 정책 상태로 변환 | MuJoCo |
| `pi05_ex_infer.py`의 동명 함수 | 위 로직 + 17-D mobile base 확장 | MuJoCo |
| `replay_trial.py::apply_rby1_action()` | 14-D action을 replay용 actuator에 매핑 | MuJoCo |

따라서 현재 상태는 다음과 같다.

```text
정책 추론
  → np.ndarray action
  → apply_action()가 정책 형식을 해석
  → 같은 함수가 MuJoCo data.ctrl에 직접 기록
```

정책 형식 해석과 backend 실행이 분리되어 있지 않으므로 SDK 또는 ROS 2를
연결하려면 `apply_action()`을 복사하여 또 다른 실행 파일을 만들게 될 가능성이
높다. 목표 구조에서는 이를 다음처럼 분리해야 한다.

```text
카메라 + robot state
        ↓
ObservationAdapter
        ↓
VLA policy
        ↓
PolicyAction14
        ↓
RBY1CommandAdapter       ← 형식, 단위, 6→7축, 제한, gripper calibration
        ↓
SafeRBY1Command
        ↓
┌────────────────┬────────────────┬─────────────────┐
│ MuJoCoBackend  │ RBY1SdkBackend │ RBY1Ros2Backend │
│ data.ctrl      │ SDK builders   │ ROS action/msg  │
└────────────────┴────────────────┴─────────────────┘
```

현재 15 Hz 절대 관절 위치 정책에는 SDK의 500 Hz UDP 실시간 API보다 SDK command
stream 또는 ROS 2 persistent trajectory stream이 적절하다.

---

## 2. 현재 저장소의 제어 계약

### 2.1 정책 관측과 action

RBY1 fine-tuned policy의 관측은 다음 세 영상과 14-D state이다.

```text
images:
  cam_high
  cam_left_wrist
  cam_right_wrist

state/action:
[
  left_arm_0, ..., left_arm_5, left_gripper,
  right_arm_0, ..., right_arm_5, right_gripper
]
```

- 관절 state: 측정된 절대 관절 위치, rad
- 관절 action: MuJoCo position actuator의 절대 목표, rad
- gripper state/action: `0=closed`, `1=open`으로 정규화
- 원본 dataset 저장 dtype: `float32`
- 실행 직전 dtype: `np.float64`
- action 차원: 14
- RBY1의 각 팔은 실제로 7-DoF지만 `arm_6`은 관측과 action에서 제외

정의와 변환은
[`pi05_infer.py`](../src/rby1_bringup/pi05_infer.py), dataset 기록 형식은
[`DATASET_COLLECTION_AND_FINETUNING_KO.md`](./DATASET_COLLECTION_AND_FINETUNING_KO.md)에
정리되어 있다.

mobile transport variant는 위 14-D 뒤에 `[base_x, base_y, base_yaw]` 절대 pose를
붙인 17-D 형식이다. 이것은 현재 MuJoCo의 kinematic base actuator용 표현이며,
실물 바퀴의 velocity 명령과 같은 의미가 아니다.

### 2.2 현재 action 적용

`rby1` action은 다음 방식으로 적용한다.

```python
left_targets = action[0:6]
left_grip = clip(action[6], 0, 1)
right_targets = action[7:13]
right_grip = clip(action[13], 0, 1)

data.ctrl[left_arm_actuator[0:6]] = left_targets
data.ctrl[right_arm_actuator[0:6]] = right_targets
data.ctrl[left_gripper] = left_grip * -0.045
data.ctrl[right_gripper] = right_grip * -0.045
```

관련 구현:

- [`pi05_infer.py::apply_action()`](../src/rby1_bringup/pi05_infer.py#L248)
- [`pi05_ex_infer.py::apply_action()`](../src/rby1_bringup/pi05_ex_infer.py#L362)
- [`replay_trial.py::apply_rby1_action()`](../scripts/replay_trial.py#L62)

현재 구현의 특징과 제약은 다음과 같다.

1. `arm_6` actuator는 갱신하지 않으므로 초기 target을 계속 유지한다.
2. gripper `0~1` clip은 있지만 팔 관절 limit, step velocity, acceleration,
   NaN/Inf, timeout 검사는 `apply_action()`에 없다.
3. 정책 차원 해석과 MuJoCo actuator 쓰기가 같은 함수에 결합되어 있다.
4. action 변환 로직이 세 파일에 중복되어 drift 가능성이 있다.
5. `rby1_mobile`의 world-frame absolute base pose는 실물의 `cmd_vel` 또는 wheel
   velocity로 변환하는 별도 controller가 필요하다.

### 2.3 현재 시간 구조

MuJoCo root model의 timestep은 `0.002 s`, 즉 500 Hz이다.

| 항목 | 현재 값 |
|---|---:|
| MuJoCo physics timestep | 0.002 s, 500 Hz |
| policy action rate | 15 Hz |
| physics steps/action | 약 33 |
| open-loop 실행 길이 | 8 actions |
| 한 chunk 실행 시간 | 약 0.533 s |

한 action target을 기록한 뒤 약 33회의 `mujoco.mj_step()` 동안 유지한다. 정책이
remote server에서 실행될 때 observation/action은 WebSocket으로 전달될 수 있지만,
최종 MuJoCo 제어는 항상 동일 프로세스의 `data.ctrl` 메모리 쓰기이다.

### 2.4 MuJoCo actuator 의미

현재 모델의 26개 기본 actuator는 모두 position actuator이다.

```text
0..1    wheel
2..7    torso
8..14   left arm
15..21  right arm
22..23  head
24..25  right/left gripper
```

정확한 순서는
[`actuators_arm.xml`](../src/rby1_description/models/rby1a/mujoco/actuators_arm.xml)에
정의되어 있다. 기본 position servo는 `kp=1500`, 일부 관절은 `kp=2000` 또는
`kp=6000`, `dampratio=1`이다.

중요하게도 MuJoCo actuator 순서는 SDK 순서와 다르다. 현재 추론 코드는 actuator
이름을 ID로 변환하므로 안전하지만, 전체 raw vector를 backend 사이에서 그대로
복사하면 좌우 팔과 바퀴 순서가 틀어진다.

---

## 3. 세 제어 방식 비교

| 구분 | 현재 MuJoCo | RBY1 SDK | RBY1 ROS 2 |
|---|---|---|---|
| 주 제어 API | `MjData.ctrl[]` | `RobotCommandBuilder` | topic/service/action |
| 관절 위치 | position actuator target | `JointPositionCommandBuilder` | `Rby1JointCommand`, `StreamJoint`, `FollowJointTrajectory` |
| Cartesian | IK 후 joint target 또는 mocap | `CartesianCommandBuilder` | `Rby1CartesianCommand` |
| 상태 | `qpos`, `qvel`, sensor | `RobotState` numpy vectors | `sensor_msgs/JointState`, `rby1_msgs/RobotState` |
| 사용자측 dtype | NumPy `float64` | NumPy/builder | ROS `float64[]`, `string[]`, `Duration` |
| 내부 직렬화 | 없음 | protobuf | ROSIDL → RMW/DDS, 이후 driver가 SDK command로 변환 |
| robot 전송 | 없음 | gRPC 또는 UDP | ROS DDS → driver → gRPC |
| 일반 command 주기 | 정책 15 Hz | gRPC once/stream, 최대 약 100 Hz | driver stream 및 action 설정값 |
| 실시간 제어 | MuJoCo physics 500 Hz | 별도 UDP callback 500 Hz | 일반 ROS graph는 hard real-time servo 용도가 아님 |
| 궤적 보간 | 사용자 코드와 MuJoCo servo | robot controller, `minimum_time`과 limit | ROS goal을 driver가 SDK builder/stream으로 변환 |
| 실행 관리 | 사용자 책임 | priority, feedback, cancel | goal/feedback/result/cancel |
| power/servo/fault | 없음 | SDK method | ROS service |

RBY1 SDK의 공식 소프트웨어 구조는 비실시간 명령을 gRPC once/stream으로, 저수준
실시간 제어를 UDP 500 Hz로 구분한다.
[`RBY1 Software Architecture`](https://rainbowrobotics.github.io/rby1-dev/software/architecture.html)

ROS 2에서 topic은 연속 data stream, service는 짧은 request/response, action은
feedback과 cancel이 필요한 장시간 작업에 사용한다.
[`ROS 2 interfaces`](https://docs.ros.org/en/humble/Concepts/Basic/Interfaces-Topics-Services-Actions.html)

---

## 4. RBY1 SDK 직접 제어

### 4.1 명령 계층

SDK의 관절 명령은 대략 다음 builder 계층을 사용한다.

```text
RobotCommandBuilder
  └─ ComponentBasedCommandBuilder
       └─ BodyComponentBasedCommandBuilder
            ├─ TorsoCommandBuilder
            ├─ RightArmCommandBuilder
            └─ LeftArmCommandBuilder
                 └─ JointPositionCommandBuilder
```

`JointPositionCommandBuilder`의 주요 입력은 다음과 같다.

| 값 | 의미 | 단위 |
|---|---|---|
| `position` | 절대 관절 목표 | rad |
| `minimum_time` | 최소 실행 시간 | s |
| `velocity_limit` | 관절 속도 제한 | rad/s |
| `acceleration_limit` | 관절 가속도 제한 | rad/s² |

팔 subsystem에 전달하는 vector는 7개여야 한다.
[`JointPositionCommandBuilder`](https://rainbowrobotics.github.io/rby1-dev/sdk/python/generated/rby1_sdk.JointPositionCommandBuilder.html)

### 4.2 once, stream, real-time

#### `robot.send_command()`

ready pose, zero pose처럼 한 개의 목표를 보내고 완료를 기다리는 point-to-point
동작에 적합하다. 15 Hz 정책 target을 매번 독립 command로 보내는 방식은 command
preemption과 긴 `minimum_time` 때문에 현재 MuJoCo 실행 의미와 달라질 수 있다.

#### `robot.create_command_stream(priority)`

목표가 지속적으로 갱신되는 VLA, teleoperation, trajectory playback에 적합한 gRPC
stream이다. 현재 정책에는 이 경로가 기본 SDK backend 후보이다.
[`SDK command stream example`](https://github.com/RainbowRobotics/rby1-sdk/blob/main/examples/python/32_command_stream.py)

#### `robot.control(callback)`

UDP 500 Hz callback에서 `ControlState`를 받고 `ControlInput`의 target,
feedback gain, feedforward torque를 반환한다. 현재 15 Hz position 정책에는 필요하지
않다. 자체 torque/position servo controller를 500 Hz에서 구현할 때 C++ 중심으로
검토한다.
[`SDK real-time example`](https://github.com/RainbowRobotics/rby1-sdk/blob/main/examples/python/34_real_time_control.py)

### 4.3 SDK 상태와 joint 순서

`RobotState`는 다음 정보를 제공한다.

- `position`, `target_position`: rad
- `velocity`, `target_velocity`: rad/s
- `torque`, `target_feedforward_torque`: Nm
- `current`: A
- FT sensor, tool flange, battery, odometry, CoM, collision 정보

[`RobotState reference`](https://rainbowrobotics.github.io/rby1-dev/sdk/cpp/state/robot-state.html)

RBY1-A의 SDK model order는 다음과 같다.

```text
right_wheel, left_wheel,
torso_0 ... torso_5,
right_arm_0 ... right_arm_6,
left_arm_0 ... left_arm_6,
head_0, head_1
```

현재 정책은 left arm부터 시작하므로 raw array index 복사를 금지하고 joint name 또는
subsystem builder를 사용해야 한다.
[`RBY1 model descriptors`](https://rainbowrobotics.github.io/rby1-dev/sdk/cpp/models/model-descriptors.html)

---

## 5. RBY1 ROS 2 제어

### 5.1 구조

ROS 2 direct driver 경로는 다음과 같다.

```text
User ROS 2 node
  → DDS/RMW topic, service, action
  → rby1_ros2_driver C++ node
  → RBY1 SDK command builder
  → gRPC
  → RBY1 robot 또는 공식 simulator
```

ROS 2는 SDK를 대체하지 않고 SDK 위에 message, discovery, lifecycle, logging,
MoveIt/TF integration 계층을 추가한다. 공식 package는 ROS 2 Humble, Ubuntu 22.04,
SDK 0.10.x 이상을 대상으로 하며 현재 beta이다.
[`RBY1 ROS 2 Driver`](https://rainbowrobotics.github.io/rby1-dev/ros2/ros2_driver.html)

별도의 MoveIt 2 경로에서는 `rby1_hardware` ros2_control plugin이 controller command를
받아 SDK gRPC stream으로 보낸다. planning이 필요하지 않은 현재 VLA는 direct driver
경로가 더 단순하다.

### 5.2 상태 topic

주요 publisher는 다음과 같다.

| Topic | Type | 내용 |
|---|---|---|
| `joint_states/torso` | `sensor_msgs/JointState` | position, velocity, torque |
| `joint_states/right_arm` | `sensor_msgs/JointState` | 오른팔 7축 상태 |
| `joint_states/left_arm` | `sensor_msgs/JointState` | 왼팔 7축 상태 |
| `joint_states/head` | `sensor_msgs/JointState` | 머리 상태 |
| `robot_state` | `rby1_msgs/RobotState` | Control Manager, brake, EMO, CoM |
| `tool_flange/left`, `right` | `rby1_msgs/ToolFlangeState` | FT/IMU/I/O |
| `odom` | `nav_msgs/Odometry` | mobile base odometry |

기본 `get_state_period=0.01 s`이면 약 100 Hz로 읽고 publish한다. policy 입력을 만들
때는 `JointState.name`을 기준으로 정렬하고, 카메라 timestamp와 동기화해야 한다.

### 5.3 단일 joint goal

`robot_joint` action type은 `rby1_msgs/action/Rby1JointCommand`이다. torso, right arm,
left arm, head를 한 goal에서 각각 지정할 수 있다.

각 component는 `rby1_msgs/msg/JointCommand`를 사용한다.

```text
string[]  joint_names
float64[] position
bool      use_impedance
bool      use_group_joint
float64   minimum_time
float64   control_hold_time
float64   velocity_limit
float64   acceleration_limit
float64[] stiffness
float64   damping_ratio
float64   torque_limit
```

- [`JointCommand.msg`](https://github.com/RainbowRobotics/rby1-ros2/blob/main/rby1_msgs/msg/JointCommand.msg)
- [`Rby1JointCommand.action`](https://github.com/RainbowRobotics/rby1-ros2/blob/main/rby1_msgs/action/Rby1JointCommand.action)

`minimum_time`의 message 기본값은 2초이므로, 이를 그대로 사용하여 15 Hz마다 새
`robot_joint` goal을 보내면 현재 MuJoCo target hold와 다른 동작이 된다. 이 action은
zero/ready pose 또는 비교적 긴 독립 motion에 사용한다.

### 5.4 persistent stream과 trajectory

현재 main source에는 다음 선택지가 있다.

- `StreamJoint`: 한 시점의 torso/양팔/head target을 persistent stream으로 갱신
- `control_msgs/action/FollowJointTrajectory`: 여러 waypoint와
  `time_from_start`를 한 goal로 전송

현재 정책은 한 번 추론할 때 8개 action을 반환하므로 8-point
`FollowJointTrajectory`로 보내는 것이 자연스럽다.

```text
joint_names = [left_arm_0 ... left_arm_6,
               right_arm_0 ... right_arm_6]

point[0].time_from_start = 1 / 15 s
...
point[7].time_from_start = 8 / 15 s
```

전송 전 `/stream_control` service의 `state=true`, `value=15.0`으로 persistent
stream을 활성화하고 종료 시 비활성화한다.
[`trajectory example`](https://github.com/RainbowRobotics/rby1-ros2/blob/main/rby1_examples/rby1_examples/10_trajectory_joint_command.py)

공식 문서 일부에는 이전 이름인 `stream_position_command`가 남아 있지만 현재 main
driver/example은 `follow_joint_trajectory`를 사용한다. 설치본에서 다음으로 실제
interface를 확인한다.

```bash
ros2 action list -t
ros2 service list -t
ros2 topic list -t
```

### 5.5 mobile base

ROS 2 mobile base 명령은 `geometry_msgs/Twist` 타입의 `cmd_vel`이다.

```text
linear.x, linear.y: m/s
angular.z: rad/s
```

따라서 현재 `rby1_mobile`의 absolute `[base_x, base_y, base_yaw]`를 직접 복사할 수
없다. odometry feedback을 받는 SE(2) pose controller가 다음을 계산해야 한다.

```text
absolute target pose
  → pose error in base frame
  → velocity/acceleration limited Twist
  → cmd_vel
```

---

## 6. 정책 14-D action을 실물 명령으로 바꾸는 규칙

### 6.1 6축 policy에서 7축 실물 팔로 확장

현재 policy는 양팔에서 `arm_6`을 예측하지 않지만 SDK arm command는 7축이다.
초기 실물 적용에서는 측정된 현재 `arm_6` 위치를 유지한다.

```python
def expand_action14(action, measured_left7, measured_right7):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (14,) or not np.all(np.isfinite(action)):
        raise ValueError("invalid RBY1 policy action")

    left7 = measured_left7.copy()
    right7 = measured_right7.copy()
    left7[:6] = action[0:6]
    right7[:6] = action[7:13]

    left_grip = float(np.clip(action[6], 0.0, 1.0))
    right_grip = float(np.clip(action[13], 0.0, 1.0))
    return left7, left_grip, right7, right_grip
```

장기적으로는 dataset과 policy를 `[L7, L-grip, R7, R-grip]` 16-D로 확장하는 것이
가장 명확하다.

### 6.2 gripper calibration

현재 MuJoCo gripper 의미는 다음과 같다.

```text
normalized 0 = closed = MuJoCo ctrl 0.0
normalized 1 = open   = MuJoCo ctrl -0.045
```

실물 gripper의 encoder tick, motor 방향, 최대 stroke와 동일하지 않다. 좌우
gripper별 calibration을 사용한다.

```text
hardware_position = closed_value
                  + normalized_opening × (open_value - closed_value)
```

RBY1 본체 SDK model의 24-DoF에는 gripper가 포함되지 않는다. 공식 문서도 gripper가
Dynamixel 계통으로 별도 제어됨을 설명한다.
[`RBY1 Python examples note`](https://rainbowrobotics.github.io/rby1-dev/examples/python_ex.html)

현재 RBY1 ROS 2 공개 joint action에도 gripper field가 없으므로 실제 장착 gripper에
맞는 Dynamixel node/action 또는 전용 driver가 필요하다.

### 6.3 안전 변환

backend로 보내기 전에 최소한 다음 순서로 검사한다.

1. shape와 dtype 검증
2. NaN/Inf 거부
3. joint name 기반 순서 변환
4. URDF/SDK joint position limit clamp 또는 command reject
5. `|q_target-q_measured| / dt` 기반 velocity 제한
6. 이전 command와 비교한 acceleration 제한
7. self-collision과 작업 공간 검사
8. action timestamp/sequence 검사
9. policy 또는 통신 timeout이면 신규 target 중단 후 안전 hold/cancel
10. gripper slew/current/stall 제한

MuJoCo backend에도 같은 제한을 적용해야 실물에서만 action이 잘리는 distribution
shift를 줄일 수 있다.

---

## 7. 권장 `RBY1CommandAdapter` 설계

### 7.1 책임 경계

`RBY1CommandAdapter`가 해야 할 일:

- policy 14-D/17-D vector의 의미와 shape 검증
- normalized gripper를 공통 명령으로 유지하거나 calibrated physical 명령으로 변환
- 6-DoF policy arm을 7-DoF hardware arm으로 확장
- left-first policy order를 named joint representation으로 변환
- joint/base limit, rate, timeout 검사
- 8-step chunk에 실행 timestamp를 부여

`RBY1CommandAdapter`가 하지 않아야 할 일:

- `data.ctrl` 직접 쓰기
- SDK `send_command()` 호출
- ROS publisher/action client 호출
- 카메라 capture
- policy inference

위 네 가지는 각각 backend 또는 observation/policy 계층의 책임이다.

### 7.2 공통 데이터 구조 예시

```python
@dataclass(frozen=True)
class PolicyAction14:
    left_q6: np.ndarray
    left_gripper: float
    right_q6: np.ndarray
    right_gripper: float


@dataclass(frozen=True)
class RBY1JointCommand:
    stamp_s: float
    left_q7: np.ndarray
    right_q7: np.ndarray
    left_gripper: float
    right_gripper: float


class RBY1Backend(Protocol):
    def read_state(self) -> RBY1State: ...
    def send_joint_command(self, command: RBY1JointCommand) -> None: ...
    def hold(self) -> None: ...
    def cancel(self) -> None: ...
```

이름은 예시이며 중요한 것은 policy 표현, 안전한 robot 표현, transport 구현을 서로
분리하는 것이다.

### 7.3 backend별 책임

#### `MuJoCoBackend`

- joint name → qpos/actuator ID map 생성
- `left_q7`, `right_q7`을 `data.ctrl`에 기록
- normalized gripper → `-0.045` position actuator로 변환
- action interval 동안 `mj_step()` 실행
- `qpos/qvel`과 renderer image 반환

#### `RBY1SdkBackend`

- connect, power, servo, Control Manager 준비
- subsystem별 `JointPositionCommandBuilder` 생성
- command stream 생성·종료
- `minimum_time`, hold time, priority 설정
- `RobotState`를 joint name 기반 공통 상태로 변환
- SDK feedback/fault/cancel 처리
- gripper는 별도 hardware driver에 위임

#### `RBY1Ros2Backend`

- `JointState`와 `RobotState` subscribe
- `/stream_control` lifecycle 처리
- 8-step chunk → `FollowJointTrajectory` 또는 매-step `StreamJoint` 변환
- action feedback/result/cancel 처리
- QoS, timestamp, stale state 검사
- camera topic과 state 동기화
- gripper는 별도 ROS 2 node/action에 위임

### 7.4 권장 파일 구조

```text
src/rby1_control/
├── types.py                 # PolicyAction14, RBY1State, RBY1JointCommand
├── command_adapter.py       # 14-D decode, 6→7축, safety/rate limits
├── observation_adapter.py   # named state + images → policy observation
├── backend.py               # RBY1Backend Protocol
├── mujoco_backend.py
├── sdk_backend.py
├── ros2_backend.py
└── gripper/
    ├── calibration.py
    ├── mujoco_gripper.py
    └── dynamixel_gripper.py
```

SDK와 ROS 2 dependency는 선택 사항이므로 backend module 안에서만 import해야 한다.
그러면 MuJoCo 개발 환경이 ROS 2나 실물 SDK 설치를 강제하지 않는다.

---

## 8. 현재 코드에서 목표 구조로 옮기는 순서

### 단계 1: 의미 보존 refactor

기존 결과를 바꾸지 않고 다음 코드를 공통 모듈로 이동한다.

- 14-D action decode
- gripper clip
- 14-D state build
- joint/actuator name 상수

기존 `pi05_infer.py`, `pi05_ex_infer.py`, `replay_trial.py`는 공통 adapter를 호출하게
한다. 같은 recorded action에 대해 refactor 전후 `data.ctrl`이 bitwise 또는 허용
오차 내에서 동일해야 한다.

### 단계 2: MuJoCo backend 분리

`apply_action()`에서 `data.ctrl` 쓰기를 `MuJoCoBackend`로 이동한다. 이 시점에도
정책 결과와 grid evaluation 결과가 달라지지 않아야 한다.

### 단계 3: hardware-ready command 추가

다음을 adapter에 추가한다.

- measured `arm_6` hold
- joint limit
- velocity/acceleration limit
- timeout/watchdog
- command sequence와 timestamp

이 제한은 MuJoCo에도 활성화하여 실물 적용 전 영향을 평가한다.

### 단계 4: 공식 RBY1 simulator에서 SDK/ROS 경로 검증

공식 simulator는 `localhost:50051`의 동일 gRPC endpoint를 제공한다. 여기서
joint order, 7축 vector, command timing, power/servo lifecycle을 확인한다.

### 단계 5: 실물 제한 시험

1. 한 팔, gripper 제외
2. 낮은 velocity/acceleration
3. 짧은 trajectory와 넓은 작업 공간
4. E-stop 준비
5. command, measured state, latency, fault를 동시 기록
6. 양팔 추가
7. calibrated gripper 추가
8. 실제 카메라와 VLA closed-loop 추가

---

## 9. 인터페이스 변환만으로 부족한 sim-to-real 항목

제어 명령의 단위와 순서를 맞추는 것은 필요조건이지만 충분조건은 아니다.

### 제어 dynamics

- MuJoCo의 높은-gain position actuator와 실물 motor controller 차이
- SDK onboard interpolation
- communication latency와 jitter
- backlash, friction, payload, cable force
- torque/current saturation

MuJoCo에서 SDK와 같은 velocity/acceleration 제한과 interpolation을 적용하고,
latency·payload·friction을 randomize하는 것이 필요하다.

### perception

- 카메라 intrinsic, extrinsic, FOV
- crop/resize와 RGB channel order
- exposure, white balance, motion blur
- sim texture와 실제 배경
- camera/state/action timestamp 정렬

현재 정책은 세 카메라에 의존하므로 perception gap이 제어 transport 차이보다 더 큰
실패 원인이 될 수 있다.

### state/action timing

반드시 아래 시점을 구분해 기록한다.

```text
t_capture     카메라 노출/프레임 시각
t_state       joint state 측정 시각
t_infer_start 정책 추론 시작
t_infer_end   action chunk 수신
t_command     backend 전송
t_apply       robot controller 적용 추정 시각
```

---

## 10. 선택 기준

| 목적 | 권장 경로 |
|---|---|
| 현재 VLA의 빠른 단일-process 실물 시험 | SDK command stream |
| 카메라, TF, rosbag2, MoveIt, 분산 시스템 통합 | ROS 2 direct driver |
| planner가 만든 collision-free trajectory 실행 | MoveIt 2 + ros2_control |
| 자체 500 Hz torque/position servo 연구 | SDK UDP real-time API |

현재 프로젝트의 최종 통합에는 ROS 2가 유리하지만, policy/action 변환과 안전 로직은
ROS node 내부가 아니라 독립 `RBY1CommandAdapter`에 두어야 한다. 그래야 같은
알고리즘과 안전 계약을 MuJoCo, SDK, ROS 2에서 공유할 수 있다.

---

## 11. 구현 전 확인 체크리스트

- [ ] policy action이 absolute position임을 모든 backend가 동일하게 해석
- [ ] left/right 및 joint 순서를 이름으로 매핑
- [ ] `arm_6` hold 정책 명시
- [ ] gripper 좌우 calibration 완료
- [ ] MuJoCo와 실물에 동일한 joint/rate limit 적용
- [ ] 15 Hz와 8-step chunk timing 유지
- [ ] state와 camera timestamp 동기화
- [ ] stale state 및 policy timeout watchdog 구현
- [ ] SDK/ROS command cancel 후 안전 hold 확인
- [ ] 공식 RBY1 simulator에서 먼저 실행
- [ ] 설치된 ROS 2 interface 이름을 introspection으로 확인
- [ ] 실물에서 command/actual/latency/fault 기록

