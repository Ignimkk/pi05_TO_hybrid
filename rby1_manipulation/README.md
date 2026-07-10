# RBY1 Manipulation — Data Collection Motion Controller

RBY1 sim에서 VLA(π0.5 등) 학습용 (obs, action) 시연 데이터를 자동으로 수집하기 위한 모션 컨트롤러. IK로 EE 궤적을 계획하고, 관절 공간에서 부드럽게 트래킹하며, LeRobot ALOHA 스키마로 저장합니다.

디렉토리 구조:
```
src/rby1_manipulation/
├── ik_utils.py                  # mink 어댑터 + 관절 인덱스 캐시
├── scene_utils.py               # 블록 랜덤화 + 자동 팔 선택
├── episode_logger.py            # LeRobot ALOHA 포맷 writer
├── follow_mocap.py              # 뷰어에서 mocap을 드래그해 팔 원격조작 (IK 검증용)
├── scenario1_single_arm.py      # 단팔 pick-place 1 에피소드
├── collect_batch.py             # 여러 에피소드 배치 수집
└── README.md                    # 본 문서
```

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

### `ik_utils.py` — IK와 관절 인덱스 캐시

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

### `scene_utils.py` — 씬 랜덤화 + 팔 자동 라우팅

| 심볼 | 역할 |
|---|---|
| `RIGHT_ARM_REACH`, `LEFT_ARM_REACH` | 각 팔의 도달 xy envelope (테이블·컨테이너 회피 반영) |
| **`set_block_pose(model, data, joint_name, xyz, quat)`** | freejoint qpos를 통해 블록 순간이동 |
| **`sample_in_reach(rng, envelope, others)`** | envelope 내 xy 샘플 (다른 블록과 `min_sep`=10cm 이상) |
| **`pick_arm_for_block(block_pos)`** | `y < -0.05 → right`, `y > 0.05 → left`, 그 외 `default` |
| **`randomize_blocks(model, data, rng)`** | 3 블록 모두 각자 팔 envelope 내로 재배치 후 forward |

### `episode_logger.py` — LeRobot ALOHA 스키마 writer

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
python src/rby1_manipulation/follow_mocap.py            # 오른팔만
python src/rby1_manipulation/follow_mocap.py --dual     # 양팔
```

뷰어에서:
1. 그리퍼 안쪽의 작은 빨간 구슬 확대 → **더블클릭** (선택)
2. **Ctrl + 우클릭 + 드래그** → 평행이동
3. **Ctrl + 좌클릭 + 드래그** → 회전

팔이 1-2초 지연으로 따라옴 (joint damping 특성). 이는 정상.

---

## 4. Single-arm 시나리오 (`scenario1_single_arm.py`)

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
python src/rby1_manipulation/collect_batch.py \
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
