# RBY1 Manipulation — Data Collection Motion Controller

RBY1 sim에서 VLA(π0.5 등) 학습용 (obs, action) 시연 데이터를 자동으로 수집하기 위한 모션 컨트롤러. IK로 EE 궤적을 계획하고, 관절 공간에서 부드럽게 트래킹하며, LeRobot ALOHA 스키마로 저장합니다.

설치 및 실행:

```bash
python -m pip install -e src/rby1_manipulation
python -m rby1_manipulation.tasks.block_pick --headless --block red
python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object apple
```

기능별 디렉토리 구조:

```
src/rby1_manipulation/
├── pyproject.toml
├── src/rby1_manipulation/
│   ├── control/                 # IK, 단팔/양팔, 모션, 모바일 베이스
│   ├── simulation/              # 블록/운반 MuJoCo 씬과 랜덤화
│   ├── planning/                # 운반 태스크 웨이포인트 계획
│   ├── evaluation/              # 성공·정지 조건
│   ├── tasks/                   # 실행 가능한 시나리오
│   ├── data/                    # 에피소드 기록과 배치 수집
│   ├── tools/                   # preview와 mocap 도구
│   └── config/                  # 패키지에 포함되는 JSON 기본값
└── tests/
```

상세한 모듈 책임, 의존 방향, 이전 파일명 매핑은
[`docs/RBY1_MANIPULATION_PACKAGE_STRUCTURE_KO.md`](../../docs/RBY1_MANIPULATION_PACKAGE_STRUCTURE_KO.md)를 참고합니다.

---

## 1. mink IK 사용 및 선택 이유

### 대안 비교

| 옵션 | 언어 | MuJoCo 통합 | 속도 | 결정론 | 멀티태스크 | 러닝커브 |
|---|---|---|---|---|---|---|
| **bio_ik** | C++ (Py wrap) | ❌ 직접 통합 | 수십 ms | ❌ GA | ⭕ | 높음 |
| **mink** ⭐ | Python | ✅ 네이티브 | 1-3 ms | ✅ QP | ⭐⭐ 우수 | 낮음 |
| placo | C++/Py | 간접 | 빠름 | ✅ | ⭐⭐ | 중간 |
| 자체 DLS Jacobian | Python + mj_jac | ✅ 네이티브 | 매우 빠름 | ✅ | ❌ 직접 짬 | 매우 낮음 |
| pyroki (JAX) | Python | 브릿지 필요 | 매우 빠름 | ✅ | ⭕ | 중간 |

### 결정 이유

- **MuJoCo 네이티브**: 우리 스택은 순수 MuJoCo + Python. bio_ik는 ROS/MoveIt 통합용이라 오버스펙 + deps 스택 무거움.
- **결정론적 QP 기반**: 데모 데이터를 대량 수집(수백~수천 에피소드)할 때 매번 다른 IK 해가 나오면 학습이 흔들림. mink의 daqp QP는 같은 입력에 같은 해.
- **멀티태스크 자연 지원**: 시나리오 2/3에서 두 EE를 동시에 제어할 handoff 국면에 `[right_task, left_task]` 리스트로 즉시 확장 가능.
- **속도**: `pip install mink` 한 줄로 설치, 1 iter ~1-3ms. 60Hz 컨트롤 루프에서 여러 inner iter를 감당.

### 핵심 사용 패턴 (요약)

```python
# 1. 모델을 mink 설정으로 감쌈
config = mink.Configuration(model)
config.update(data.qpos)

# 2. 태스크 정의
task = mink.FrameTask("right_ee", "site",
                     position_cost=100.0, orientation_cost=10.0, lm_damping=0.001)
task.set_target(mink.SE3.from_rotation_and_translation(rot, pos))

posture = mink.PostureTask(model, cost=1e-4)   # 다른 관절 유지
posture.set_target_from_configuration(config)

# 3. IK 풀기 → 관절 속도 반환
vel = mink.solve_ik(config, [task, posture], dt=1e-2, solver="daqp", damping=1e-4)

# 4. arm DoF 외는 zeroing (필수, 아래 함정 참조)
vel = vel * arm_dof_mask
config.integrate_inplace(vel, dt)
```

### 셋업 중 발견한 3가지 함정

**함정 1 — mink는 기본적으로 model의 모든 DoF를 씀**
> RBY1은 base planar 3-DoF + torso 6-DoF + 각 팔 7-DoF. 마스크 없이 돌리면 IK가 "base를 6cm 옆으로 옮기고 arm은 조금만 움직여" 하는 해를 잡음. 순수 kinematic으론 수렴하는데 실제 sim에선 base가 wheel 액추에이터로 구동되지 EE 위치로 순간이동 못 하므로 EE가 목표에 도달 못 함.
>
> **대응**: `arm_dof_mask` (`build_dof_mask`) 만들어서 IK velocity를 arm DoF만 통과시킴.

**함정 2 — joint damping=50 이 커서 트래킹 느림**
> 위치 액추에이터가 target에 도달하는 데 1-2 초. 실시간 teleop엔 답답하지만 스크립트 waypoint 방식(각 waypoint 후 wait) 엔 문제 없음.

**함정 3 — 매 tick config.update(data.qpos) 하면 진행 안 됨**
> `config.update`는 config를 지연된 sim 상태로 되돌리는 리셋 연산. IK가 "앞으로 나갔다가" 지연 상태로 리셋되기를 반복해서 오차가 안 줄어듦.
>
> **대응**: 컨트롤 tick당 IK **inner iteration 10회** 정도 돌려 config가 수렴한 후에 ctrl로 밀어냄. inner loop 안에서는 config.update 안 함.

---

## 2. 유틸리티 함수

### `control/ik.py` — IK와 관절 인덱스 캐시

| 심볼 | 역할 |
|---|---|
| `RIGHT_ARM_JOINTS`, `LEFT_ARM_JOINTS` | 7-DoF 관절명 리스트 |
| `RIGHT_ARM_ACTS`, `LEFT_ARM_ACTS` | 대응 액추에이터명 리스트 |
| `RIGHT_EE_SITE`, `LEFT_EE_SITE` | IK 기준 EE 사이트명 |
| `GRIPPER_R_ACT`, `GRIPPER_L_ACT` | 그리퍼 액추에이터명 |
| `GRIPPER_OPEN=-0.05`, `GRIPPER_CLOSED=0.0` | ctrl 범위 상수 |
| **`ArmHandles`** dataclass | 팔 하나의 qidx/dofidx/aid/gripper 인덱스를 한 곳에 캐시 |
| **`right_arm_handles(model)`, `left_arm_handles(model)`** | ArmHandles 생성 팩토리 |
| **`build_dof_mask(model, joint_names)`** | `nv` 크기 bool 마스크 → 지정 관절 DoF만 True |
| **`solve_kinematic_ik(...)`** | 오프라인 kinematic IK: seed_qpos + target_pose → 수렴한 qpos 벡터 (arm DoF만 변화) |
| **`site_pose(data, model, site_name)`** | 사이트 세계 pose → `mink.SE3` |
| **`se3_at(position, orientation)`** | 편의 생성자 |
| **`set_arm_ctrl(data, arm, target_qpos_all)`** | 전체 qpos 벡터에서 arm 관절만 뽑아 액추에이터에 설정 |
| **`set_gripper(data, arm, 'open'/'close')`** | 그리퍼 ctrl 세팅 |

### `simulation/common.py` — 씬 랜덤화 + 팔 자동 라우팅

| 심볼 | 역할 |
|---|---|
| `RIGHT_ARM_REACH`, `LEFT_ARM_REACH` | 각 팔의 도달 xy envelope (테이블·컨테이너 회피 반영) |
| **`set_block_pose(model, data, joint_name, xyz, quat)`** | freejoint qpos를 통해 블록 순간이동 |
| **`sample_in_reach(rng, envelope, others)`** | envelope 내 xy 샘플 (다른 블록과 `min_sep`=10cm 이상) |
| **`pick_arm_for_block(block_pos)`** | `y < -0.05 → right`, `y > 0.05 → left`, 그 외 `default` |
| **`randomize_blocks(model, data, rng)`** | 3 블록 모두 각자 팔 envelope 내로 재배치 후 forward |

### `data/episode.py` — LeRobot ALOHA 스키마 writer

| 심볼 | 역할 |
|---|---|
| `CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")` | ALOHA 규격 카메라 이름 |
| `CHUNK_SIZE = 1000` | LeRobot 관례 |
| **`Frame(state, action, images, timestamp, frame_index)`** | 프레임 하나 컨테이너 (dataclass) |
| **`EpisodeBuffer(episode_index, task, frames=[])`** | 에피소드 하나 (in-memory) |
| **`LeRobotWriter(root, fps, image_wh)`** | 스트리밍 writer. `new_episode(task)` → 프레임 append → `save_episode(ep)` |
| `.finalize()` | `meta/info.json`, `episodes.jsonl`, `tasks.jsonl` 생성 |

출력 규격:
```
<root>/
  meta/info.json           # fps, cameras, feature dtypes
  meta/episodes.jsonl      # 에피소드별 length + task
  meta/tasks.jsonl         # task 문자열 index
  data/chunk-000/episode_XXXXXX.parquet    # state(14), action(14), timestamps 등
  videos/chunk-000/observation.images.<cam>/episode_XXXXXX.mp4
```

**State/Action 레이아웃 (14차원, ALOHA 규격)**:
```
[left_arm_0, ..., left_arm_5, left_gripper,  right_arm_0, ..., right_arm_5, right_gripper]
   6                            1              6                              1
```
> RBY1의 7-DoF 팔에서 마지막 wrist 관절(`arm_6`)은 잘라내고 6-DoF만 저장 (ALOHA 규격이 6+1임).

---

## 3. `follow_mocap.py` — 인터랙티브 IK 검증

목적: mink IK와 관절 트래킹이 실제로 작동하는지 뷰어에서 즉시 확인. 데이터 수집 자체엔 안 쓰지만 시나리오 개발 전 감 잡기용.

### 아키텍처

```
┌───────────────────────────────┐
│ MuJoCo viewer (사람)           │
│ - mocap 구를 Ctrl+drag         │
│ - data.mocap_pos/quat 변화     │
└─────────────┬─────────────────┘
              ▼
┌───────────────────────────────┐
│ 컨트롤 루프 (60Hz)             │
│ 1. mocap → FrameTask.set_target│
│ 2. config.update(data.qpos)    │
│ 3. inner IK × IK_INNER_ITERS   │
│    - solve_ik → vel            │
│    - vel *= arm_mask           │
│    - config.integrate(vel, dt) │
│ 4. data.ctrl[arm] = config.q   │
│ 5. mj_step × sim_steps         │
│ 6. viewer.sync()               │
└───────────────────────────────┘
```

### 핵심 결정사항

- **arm-DoF 마스크**: 함정 1 대응. `--dual`일 땐 양팔 관절 모두 True, 그 외는 오른팔만.
- **inner IK 반복**: 함정 3 대응. 매 tick 10회 iter 후에야 ctrl 설정.
- **mocap 자동 스냅**: 시작 시 `snap_mocap_to_site`가 mocap 구를 현재 EE 위치에 붙임 → 첫 tick에 팔이 튀지 않음.
- **뷰어 마커 자동 활성화**: `viewer.opt.geomgroup[3] = 1`, `sitegroup[4] = 1` — 다른 스크립트에서 숨겨둔 mocap 마커를 이 스크립트에선 자동으로 켬.

### 실행

```bash
python -m rby1_manipulation.tools.follow_mocap            # 오른팔만
python -m rby1_manipulation.tools.follow_mocap --dual     # 양팔
```

뷰어에서:
1. 그리퍼 안쪽의 작은 빨간 구슬 확대 → **더블클릭** (선택)
2. **Ctrl + 우클릭 + 드래그** → 평행이동
3. **Ctrl + 좌클릭 + 드래그** → 회전

팔이 1-2초 지연으로 따라옴 (joint damping 특성). 이는 정상.

---

## 4. Single-arm 시나리오 (`rby1_manipulation.tasks.block_pick`)

### 태스크 정의
`R/G/B 블록 하나를 (자동으로 선택된) 팔로 pick → 갈색 컨테이너에 place`

**언어 프롬프트**:
```
pick up the {red|green|blue} block and put it in the brown box
```

### Waypoint 시퀀스 (8단계)

| # | label | EE 위치 | gripper | 지속시간 | 대기 |
|---|---|---|---|---|---|
| 1 | approach | 블록 위 10cm hover | open | 1.5s | 0.4s |
| 2 | descend | 블록 위 2cm | open | 1.2s | 0.4s |
| 3 | grasp | (동일 위치) | **close** | 0.1s | 0.8s |
| 4 | lift | 블록 위 15cm | hold | 1.2s | 0.4s |
| 5 | carry | 컨테이너 위 15cm | hold | 1.8s | 0.4s |
| 6 | descend2 | 컨테이너 위 8cm | hold | 1.0s | 0.4s |
| 7 | release | (동일 위치) | **open** | 0.1s | 0.6s |
| 8 | retract | 컨테이너 위 20cm | open | 1.2s | 0.4s |

**Orientation**: 그리퍼는 항상 초기 keyframe의 EE 방향 (아래 향함) 유지.

### 실행 구조

```
main()
├── mj_resetDataKeyframe(teleop)
├── if --random: randomize_blocks(model, data, rng)     # scene_utils
├── settle_scene()                                       # 물리 안정화 1.5s
├── if --arm=auto: pick_arm_for_block(block_pos)         # scene_utils
├── ee_down = site_pose(data, model, arm.ee_site).rotation()
├── waypoints = make_waypoints(block_pos, container_pos, ee_down)
├── execute_waypoints(model, data, arm, arm_mask, waypoints, on_step)
│   └── for wp in waypoints:
│       ├── q_target = solve_kinematic_ik(...)          # ik_utils
│       ├── set_gripper(wp.gripper)
│       ├── 관절공간 선형 ramp × ramp_steps
│       │   └── on_step()  ← 프레임 로깅
│       └── hold × hold_steps
│           └── on_step()
├── check_success(model, data, block_name, container_pos)
└── if --log-dataset and success: writer.save_episode(); writer.finalize()
```

### CLI 옵션

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--arm` | `auto` | `auto` / `right` / `left` |
| `--block` | `red` | `red` / `green` / `blue` |
| `--random` | off | 블록 xy를 각 팔 envelope 내로 재샘플 |
| `--seed N` | None | `--random`과 함께 |
| `--headless` | off | 뷰어 없이 실행 |
| `--record path.mp4` | None | 3인칭 뷰 mp4 저장 |
| `--log-dataset root` | None | LeRobot 데이터셋 root에 저장 (성공한 에피소드만) |
| `--log-fps N` | 15 | 데이터셋 프레임 레이트 |

### 자동 팔 라우팅 (auto)

```python
if block.y < -0.05:  # 로봇 오른쪽
    arm = right
elif block.y > +0.05: # 로봇 왼쪽
    arm = left
else:                 # 중앙
    arm = right (default)
```

### 도달 범위 인식 랜덤화

`--random`은 각 팔의 envelope 내에서 xy를 재샘플:
- Right envelope: `x ∈ [0.45, 0.60], y ∈ [-0.32, -0.15]`
- Left envelope:  `x ∈ [0.45, 0.60], y ∈ [+0.15, +0.32]`

경계 근처는 실패 가능성 존재 → 실제 성공률 **85-90%**.

### 배치 수집 (`collect_batch.py`)

```bash
python -m rby1_manipulation.data.collect_batch \
    --root ~/dev_ws/vla/pi0_TO_ws/data/rby1_scenario1 \
    --colors red green blue \
    --n-per-color 20 \
    --seed-start 0 \
    --log-fps 15
```

- 하나의 `LeRobotWriter` 인스턴스가 모든 색깔·seed 조합 진행
- 성공한 에피소드만 `episode_XXXXXX.parquet` 로 append (연속 index)
- 실패 에피소드는 저장 안 됨 → 정확한 시연 데이터만 남음
- ~2.8초/에피소드 (헤드리스), 100 에피소드 기준 ~5분

### 성공 판정

```python
def check_success(model, data, block_name, container_pos):
    p = block world pos
    inside_xy = |p[0]-container.x| < 0.11  AND  |p[1]-container.y| < 0.11
    above_table = p[2] > 0.82 - 0.01
    return inside_xy and above_table
```

컨테이너 xy 내 + 테이블 위(밖에 떨어지지 않음).

---

## 개발 히스토리 요약

| 단계 | 결과 |
|---|---|
| mink 스모크 테스트 | 5cm 이동 요청 → 3mm 오차. **채택.** |
| follow_mocap.py 첫 시도 | EE가 안 따라옴. **원인**: mink가 base/torso DoF까지 씀. |
| follow_mocap.py + DoF mask | EE 트래킹 시작하지만 느림. **원인**: joint damping=50 + 매 tick config.update. |
| follow_mocap.py + inner iter | 정상 동작. 1-2초 지연은 물리 특성. |
| scenario1 첫 실행 | 컨테이너 x=0.7 시절 err 68mm. 블록은 배럴리 안착. |
| scenario1 + container x=0.6 | err 25mm. 안정 성공. |
| scenario1 + auto-arm + randomize | 85-90% 성공률. 데이터 다양성 확보. |
| LeRobot logger 통합 | 3 카메라 + 14dim state/action 정상 저장. |
| collect_batch 배치 러너 | 여러 에피소드 연속 저장. episode_index 연속 유지. |

## 다음 단계

- 시나리오 2: 왼팔 pick → **오른팔 handoff** → 오른팔 place  
- 시나리오 3: 오른팔 pick → **왼팔 handoff** → 왼팔 place  
- handoff은 두 EE를 같은 위치 + 반대 방향으로 맞추는 duo-frame IK 국면 + gripper 인수인계 타이밍 조율이 핵심.

---

# 모바일 양팔 운반 (crate transport) 시나리오

> 변경 내역 전체 정리는 [`docs/TRANSPORT_SCENARIO_KO.md`](../../docs/TRANSPORT_SCENARIO_KO.md)를 보세요.

기존 블록 pick-and-place 위에 **모바일 매니퓰레이션 VLA 시나리오**를 추가한 것입니다.
기존 파이프라인은 한 줄도 바꾸지 않았습니다: `model.xml`, `scenario1/2/3`,
`collect_dataset.py`, `preview_block_grid.py`는 그대로이고, 새 씬은 별도 root
`model_transport.xml`에서 동작합니다.

## 시나리오

| 스크립트 | 내용 |
|---|---|
| `scenario_transport_crate.py` | 양팔로 크레이트 파지 → 리프트 → 베이스 주행 → 선반 배치 |
| `scenario_transport_load_and_carry.py` | 한 손으로 과일 4종 중 하나를 파지 → 크레이트에 담기 → 위 시나리오 수행 |

```bash
python -m rby1_manipulation.tasks.transport_crate --headless
python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object apple
python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object banana
python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object orange
python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object pear
python -m rby1_manipulation.tasks.transport_crate --headless --random --seed 3      # 도메인 랜덤화
python -m rby1_manipulation.tasks.transport_crate --headless --log-dataset /path/ds  # 17-D LeRobot 기록
```

## 새 파일

```
rby1_description/models/rby1a/mujoco/
├── defaults_common.xml          # 공용 <default> (기존 model.xml에서 추출) + prop_visual/prop_collision
├── actuators_arm.xml            # 기존 26개 액추에이터 (ctrl 0~25, 순서가 인터페이스)
├── actuators_base.xml           # base_x/y/yaw position 액추에이터 (ctrl 26~28)
├── scenes/scene_transport.xml   # 사무실 공간 + 테이블 + 크레이트 + 과일 4종 + 3단 선반
├── scenes/office_assets.xml     # 사무실 바닥/벽 재질
├── model_transport.xml          # 새 root (기본)
└── model_transport_wheels.xml   # 실험용 차동구동 variant

rby1_manipulation/
├── transport_layout.json        # 크레이트/물체/선반/도킹 포즈 설정
├── transport_scene.py           # config·리셋·랜덤화·17-D state/action, --self-check / --reach-report
├── transport_plan.py            # 두 시나리오가 공유하는 웨이포인트 빌더
├── bimanual_ik.py               # 양팔 IK 래퍼 + BiWaypoint 실행기
├── motion_utils.py              # 램프/홀드/adaptive close (양팔·베이스)
├── success_checks.py            # check_grasp / object_in_crate / crate_on_shelf / StopCheck
├── episode_recording.py         # mp4 + LeRobot 프레임 캡처
├── preview_transport_layout.py  # 레이아웃 뷰어/튜너
└── wheel_drive.py               # 실험용 차동구동 컨트롤러
```

## 인터페이스

- **카메라·관측**: `zed_left`/`wrist_cam_l`/`wrist_cam_r` → `cam_high`/`cam_left_wrist`/`cam_right_wrist`,
  224×224 그대로. **변경 없음.**
- **state/action**: 17-D `[L arm 0..5, L grip, R arm 0..5, R grip, base_x, base_y, base_yaw]`.
  앞 14차원은 기존과 완전히 동일(그리퍼 정규화 `abs(qpos)/0.045` 포함)합니다.
  `episode_logger.LeRobotWriter(..., schema="rby1_17_mobile")`로 opt-in하며,
  기본값 `rby1_14`는 불변이라 기존 수집 스크립트는 그대로 동작합니다.
- **추론**: `pi05_ex_infer.py --model rby1_mobile` (obs/action format `rby1_mobile`).
  `ARM_DIMS`는 그대로 두어 BJ/IJ/CD 지표를 기존 실험과 비교할 수 있고,
  베이스는 `BASE_DIMS = [14,15,16]`로 따로 봅니다.

## 검증 — 실험별 명령어

모두 `src/rby1_manipulation/`에서 실행하고, `python`은 `/home/mk/venv/pi0_TO_env/bin/python`
(mink가 설치된 유일한 env)입니다.

### A. 씬이 제대로 보이는지 (눈으로 확인)

| 실험 | 명령어 | 합격 기준 |
|---|---|---|
| 씬 뷰어 | `python preview_transport_layout.py` | 사무실 방 안에 크레이트·손잡이·사과·바나나·오렌지·배·3단 선반이 모두 보임. 최상단(목표) 판은 청록색 |
| 좌표 리포트 | `python preview_transport_layout.py --report-only` | 크레이트 정착 z=0.880, 손잡이 z=0.965, 베이스-선반 여유 0.120 m |

> 뷰어에서 group 3(충돌 지오메트리)을 보려면 키보드 `3`을 누르세요. 기본은 숨김이며,
> 씬의 모든 물체는 group 2 시각 geom을 따로 갖고 있습니다.
>
> 3인칭 카메라는 **방 안에** 두어야 합니다. 실내가 x ∈ [-2.9, 3.5], y ∈ [-3.7, 3.1]이므로
> 거리 5 m를 넘기면 벽 바깥으로 나가 벽면만 찍힙니다
> (`episode_recording.RECORD_DISTANCE = 3.0`).

### B. 모델 정합성 (빠름, 각 10초~2분)

| 실험 | 명령어 | 합격 기준 |
|---|---|---|
| 모델 불변식 | `python transport_scene.py --self-check` | `SELF-CHECK PASSED` |
| IK 도달성 | `python transport_scene.py --reach-report` | `worst residual 2.0 mm` / `PASSED` |

### C. 시나리오 동작 (각 2~4분)

| 실험 | 명령어 | 합격 기준 |
|---|---|---|
| 시나리오 1 | `python -m rby1_manipulation.tasks.transport_crate --headless` | `SUCCESS = True`, `xy_err` ≤ 0.01, `tilt=0.0deg` |
| 시나리오 2 (사과) | `python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object apple` | 위 + `apple still in crate = True` |
| 시나리오 2 (바나나) | `python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object banana` | 위 + `banana still in crate = True` |
| 시나리오 2 (오렌지) | `python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object orange` | 위 + `orange still in crate = True` |
| 시나리오 2 (배) | `python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object pear` | 위 + `pear still in crate = True` |
| 영상으로 확인 | `python -m rby1_manipulation.tasks.transport_crate --headless --record /tmp/crate.mp4` | mp4 생성 |

### D. 데이터셋 (17-D 신규 / 14-D 불변)

| 실험 | 명령어 | 합격 기준 |
|---|---|---|
| 17-D 수집 | `python -m rby1_manipulation.tasks.transport_crate --headless --log-dataset /tmp/ds17` | `episode 0 (...) -> /tmp/ds17` |
| 스키마 확인 | `python ../../scripts/validate_dataset.py --dataset /tmp/ds17` | `state shape=(N, 17)`, `done.` |
| 14-D 회귀 | `python -m rby1_manipulation.tasks.block_pick --headless --block red --log-dataset /tmp/ds14` | parquet이 `[14]`, `robot_type="rby1"` |

### D-1. 고정 베이스 과일 적재·상자 들기 데이터셋 (14-D)

새 collector는 wheel·베이스 이동·선반 배치·장애물을 사용하지 않습니다.
동작과 waypoint 사이 대기는 검증된 기본 `1.25x` timing을 사용하며,
`--speed-scale 1.0`으로 기존 timing을 재현할 수 있습니다.

```bash
# 1,200개 균형 schedule 확인
python -m rby1_manipulation.data.collect_transport_dataset \
  --output-dir /data/rby1_transport_14d --dry-run

# 성공 episode 1,200개 수집
python -m rby1_manipulation.data.collect_transport_dataset \
  --output-dir /data/rby1_transport_14d
```

- `pack_only`, `lift_only`, `pack_and_lift`: 각 400개
- 16개 fruit grid layout: 각 75개
- 과일 종류, 적재 순서, 좌우 table slot 균형
- `lift_only`: 사전 적재 0~4개를 각 80개
- state/action: 기존 block과 같은 `rby1_14`

상세 설계와 단일 episode 명령은
[`docs/TRANSPORT_DATASET_14D_KO.md`](docs/TRANSPORT_DATASET_14D_KO.md)에 있습니다.

### E. 기존 파이프라인 회귀

| 실험 | 명령어 | 합격 기준 |
|---|---|---|
| 블록 시나리오 | `python -m rby1_manipulation.tasks.block_pick --headless --block red` | `SUCCESS = True` |
| model.xml no-op | 아래 스니펫 | `identical: True` |

```bash
cd ../  # src/
git show HEAD:rby1_description/models/rby1a/mujoco/model.xml \
  > rby1_description/models/rby1a/mujoco/_orig.xml
python -c "
import mujoco
D='rby1_description/models/rby1a/mujoco/'
f=lambda p:(lambda m:(m.nq,m.nv,m.nu,m.actuator_gainprm.round(9).tolist(),
  m.actuator_biasprm.round(9).tolist(),m.body_mass.round(9).tolist(),
  m.key_qpos.round(9).tolist()))(mujoco.MjModel.from_xml_path(D+p))
print('identical:', f('_orig.xml')==f('model.xml'))"
rm rby1_description/models/rby1a/mujoco/_orig.xml
```

### F. 랜덤화 견고성 (시드당 2~4분)

아래 사과/바나나 성공률은 과일 mesh 및 4종 확장 전의 과거 측정값입니다. 현재 4종 배치기는
100개 seed에서 최소 중심 간격 0.080 m를 통과했으며, 전체 task 성공률 sweep은 다시 측정해야 합니다.

| 실험 | 명령어 | 현재 결과 |
|---|---|---|
| 시나리오 1 | `for s in $(seq 0 19); do python -m rby1_manipulation.tasks.transport_crate --headless --random --seed $s \| grep -q "SUCCESS = True" && echo "$s PASS" \|\| echo "$s FAIL"; done` | **19/20** |
| 시나리오 2 (바나나) | 위에서 스크립트만 `scenario_transport_load_and_carry.py --object banana`로 교체 | 8/10 |
| 시나리오 2 (사과) | 위에서 `--object apple` | **3/10 — 미해결** |
| 시나리오 2 (오렌지/배) | 각각 `--object orange`, `--object pear` | random pose 전체 sweep 미측정 |

### G. wheel 모드 (실험용, 성공 기대 안 함)

```bash
python -m rby1_manipulation.tasks.transport_crate --headless --base-mode wheel
```
배너 출력 + NaN 없이 완주하면 통과입니다. `SUCCESS = False`가 정상입니다.

## 이 씬에서 반드시 알아야 할 것들 (실측)

1. **베이스는 바닥에 박혀 있다.** `base`에 z DoF가 없고 충돌 메쉬 하단이 z=-0.003이라
   바닥을 2.6 mm 관통 → 정상상태 수직항력 **248,943 N**. `model_transport.xml`의
   `<contact><exclude>`(world↔base/wheel_r/wheel_l)가 없으면 어떤 gain으로도 움직이지 않습니다.
2. **`impratio="10"`이 양팔 파지를 가능하게 한다.** 기본값 1에서는 모든 접촉이 마찰 원뿔
   안쪽인데도 크레이트가 초당 5 mm씩 미끄러져 4초 만에 떨어집니다. 10에서는 10초간 3.8 mm.
3. **`class="collision"`을 씬 물체에 쓰면 안 된다.** `conaffinity=0`이라 로봇 손가락과
   접촉 자체가 계산되지 않습니다. `prop_collision`(contype=1 conaffinity=1)을 쓰세요.
4. **`GRASP_PAD_OFFSET = 0.003`** — 패드 중심은 EE 사이트와 거의 일치합니다.
   0.0244로 잡으면 봉이 패드 아래 가장자리에 걸려 1초 만에 빠집니다.
5. **리프트는 `carry.clear_z` 기준 절대 높이 + 폐루프 트림.** 팔이 하중으로 ~70 mm 처지므로
   개루프 리프트로는 크레이트가 선반 상단 판보다 낮아져 주행 중 들이받습니다.
6. **손잡이 위를 지나는 경로.** 손가락이 패드 중심보다 ~40 mm 아래로 뻗으므로,
   소형 물체 접근·운반 경로는 손잡이 상단 + 0.14 m로 넘어가야 합니다.
7. **소형 물체는 |y| ≥ 0.29에 스폰.** 손잡이 자체는 |y|=0.21까지지만 팔뚝이 훨씬 넓어
   |y| ≤ 0.28에서는 하강 시 손잡이에 걸려 50~80 mm 오차가 납니다.
8. **바나나는 시각=캡슐, 충돌=박스.** 수평 원기둥은 평면 패드 사이에서 튕겨 나갑니다.
9. **선반 목표는 최상단(위가 열린 단)만 가능.** 그리퍼+손목이 크레이트 위로 ~0.25 m
   튀어나와, 위에 판이 있는 단에는 넣을 수 없습니다. 하단 판은 distractor입니다.
10. **모델은 절대경로로 로드.** 상대경로면 `rby1.xml`의 WHEEL geoms 중복 include에서
    XML 오류가 납니다(mujoco 3.10.0 기준).

## 그리퍼 개도 제어

그리퍼는 열림/닫힘 두 값이 아니라 **원하는 만큼** 열 수 있습니다.

```python
from rby1_manipulation.control.ik import (
    gripper_width_from_qpos,
    set_gripper,
    set_gripper_width,
)

set_gripper(data, arm, "open")        # 기존 인터페이스 그대로 (= 86.4 mm)
set_gripper(data, arm, "close")
set_gripper(data, arm, 0.6)           # 전체 스트로크의 60 %
set_gripper_width(data, arm, 0.030)   # 패드 간격 30 mm 지정
gripper_width_from_qpos(data.qpos[arm.gripper_qidx])   # 현재 간격 읽기
```

- 변환식은 실측입니다: `gap(q) = 2|q| - 0.0036`. 조인트 한계에서 최대 **96.4 mm**,
  `GRIPPER_OPEN`(-0.045)이 86.4 mm입니다. 명령한 폭과 실측 폭이 0.0 mm 오차로 일치합니다.
- `BiWaypoint.gripper`도 `"open"/"close"/"hold"` 외에 **float**를 받습니다.
- 시나리오 CLI: `--grip-open 0.6` (접근·릴리스 시 개도). 기본 1.0.
- `set_gripper`의 기존 3-way 호출은 그대로 동작하므로 기존 코드 영향 없음.

## wheel 모드

`--base-mode wheel`은 `model_transport_wheels.xml` + `actuators_arm_wheeldrive.xml`을
로드해 두 바퀴로 베이스를 구동합니다.

**직진은 정상 동작합니다** — 바퀴 -1.90 rad/s에 베이스 +0.201 m/s로 유효 구름 반경
0.1004 m, 미끄러짐 0의 완전한 no-slip 구름입니다.

이렇게 되기까지 고친 것:
- 바퀴 액추에이터의 `ctrlrange`가 `[-3.14, 3.14]`였습니다. 조인트는 `limited="false"`인데
  `inheritrange="1"`이 기본 클래스의 range를 가져간 탓으로, **바퀴가 반 바퀴 이상 돌 수
  없어 최대 ±0.31 m만 이동 가능**했습니다. → 전용 velocity 액추에이터로 분리.
- `WHEEL_RADIUS`가 0.0602로 40 % 낮게 잘못 잡혀 있었습니다 → 실측 **0.1004**.
- 부호 규약 실측 확정: 전진 = **음의** 바퀴 속도, `L+ / R-` = +yaw.

**제자리 회전은 아직 안 됩니다.** 명령 0.758 rad/s 대비 실제 0.13 rad/s이고, 원인은
바퀴 스톨입니다(yaw는 실제 바퀴 속도의 구름 예측과 정확히 일치 = 미끄러짐 0).
근본 원인은 `base`에 z 자유도가 없어 접촉 수직력이 바퀴당 **28 kN**(실제 무게의 50배)이
되는 것으로, 접촉 강성·armature·damping을 낮추는 시도는 모두 발산(NaN)했습니다.
제대로 고치려면 `rby1.xml`의 `base`에 수직 자유도가 필요한데, 이 파일은 블록 파이프라인과
공유되며 `nq`가 바뀌어 기존 keyframe·데이터셋 인덱싱이 깨집니다.

**데이터 수집에는 kinematic 모드(기본)를 사용하세요** — 베이스 포즈가 정확히 재현됩니다.

정적·동적 장애물 profile, 충돌/최소거리 감시, wheel 사전점검과 현재 motion-test
상태는 [`docs/RBY1_OBSTACLE_WHEEL_READINESS_KO.md`](../../docs/RBY1_OBSTACLE_WHEEL_READINESS_KO.md)에
정리되어 있습니다. 장애물은 평가 전용이며 학습 데이터 수집에는 사용할 수 없습니다.
