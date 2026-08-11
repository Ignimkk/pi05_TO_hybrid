# RB-Y1 모바일 양팔 운반 시나리오 — 변경 내역 정리

기존 RB-Y1 블록 pick-and-place 환경 위에 **모바일 매니퓰레이션 VLA 시나리오** 2종을
추가하면서 수정·추가한 내용 전부를 정리한 문서입니다.

- 시나리오 1: 양팔로 크레이트 파지 → 리프트 → 베이스 주행 → 선반 배치
- 시나리오 2: 한 손으로 사과/바나나/오렌지/배 파지 → 크레이트에 담기 → 시나리오 1 수행

실행 명령어는 [`src/rby1_manipulation/README.md`](../src/rby1_manipulation/README.md)의
"검증 — 실험별 명령어" 절(A~G)에 정리되어 있습니다.

---

## 1. 최우선 원칙: 기존 파이프라인 무손상

블록 데이터셋(`rby1_dataset_v1`)과 `pi05_rby1_lora` 체크포인트, grid 평가의 재현성을
깨지 않는 것이 모든 설계 결정의 상위 제약이었습니다. 그래서:

- 새 씬은 **별도 root** `model_transport.xml`에서 동작합니다. `model.xml`은 `<default>`와
  `<actuator>`를 include로 빼는 리팩터만 했고, **컴파일 결과가 git 원본과 완전히 동일**함을
  A/B로 검증했습니다(`nq/nv/nu`, 액추에이터 이름·trnid, gainprm/biasprm, ctrlrange,
  geom group/contype/conaffinity, body_mass/inertia, keyframe 전부 일치).
- 블록·handoff 동작과 `rby1.xml`의 모델 계약은 유지합니다. Python 구현은 이후
  `rby1_manipulation` 설치형 패키지로 이동했으며 실행 경로는
  `rby1_manipulation.tasks.*`로 통일했습니다.
- 공유 코드는 `control`, `simulation`, `planning`, `evaluation`, `data`로 분리했습니다.
  상세 매핑은 [패키지 구조 문서](RBY1_MANIPULATION_PACKAGE_STRUCTURE_KO.md)에 있습니다.

---

## 2. 한눈에 보는 변경 목록

### 신규 MJCF — `src/rby1_description/models/rby1a/mujoco/`

| 파일 | 줄수 | 내용 |
|---|---:|---|
| `defaults_common.xml` | 43 | `model.xml`에서 추출한 공용 `<default>` + 신규 `prop_visual` / `prop_collision` 클래스 |
| `actuators_arm.xml` | 39 | 기존 26개 액추에이터를 그대로 이동. **ctrl 0~25 순서가 인터페이스** |
| `actuators_base.xml` | 25 | `base_x_act` / `base_y_act` / `base_yaw_act` (ctrl 26~28) |
| `actuators_arm_wheeldrive.xml` | 47 | 위와 동일 순서, 바퀴만 velocity 액추에이터 |
| `scenes/scene_transport.xml` | — | **사무실 공간** + 테이블 + 크레이트 + 과일 4종 + 3단 선반 |
| `scenes/office_assets.xml` | 20 | 사무실 바닥/벽/트림/유리 재질 |
| `scenes/transport_prop_assets.xml` | — | 과일 4종의 visual·collision OBJ와 material 등록 |
| `assets/transport_props/*.obj` | 17개 | 절차적으로 생성한 과일 visual mesh와 convex collision decomposition |
| `model_transport.xml` | 80 | **기본 root** (kinematic 베이스) |
| `model_transport_wheels.xml` | 68 | 휠 구동 variant |

### Python 패키지 — `src/rby1_manipulation/src/rby1_manipulation/`

| 파일 | 줄수 | 역할 |
|---|---:|---|
| `config/transport_layout.json` | 크레이트/물체/선반/도킹 포즈 설정 |
| `simulation/transport_scene.py` | config 로드·검증, 리셋·랜덤화, 베이스 핸들, 17-D state/action |
| `planning/transport.py` | 두 시나리오가 공유하는 웨이포인트 빌더 |
| `control/bimanual.py` | 양팔 IK 래퍼, `BiWaypoint`, 웨이포인트 실행기 |
| `control/motion.py` | 램프/홀드/adaptive close |
| `evaluation/transport.py` | grasp/in-crate/on-shelf/stop 판정 |
| `tasks/transport_crate.py` | 시나리오 1 |
| `tasks/transport_load_and_carry.py` | 시나리오 2 |
| `data/recording.py` | mp4 + LeRobot 프레임 캡처 |
| `tools/preview_transport_layout.py` | 레이아웃 뷰어/튜너 |
| `control/mobile_base.py` | 휠 차동구동 컨트롤러 |

### 수정 파일 (6개, +384 / −75줄)

| 파일 | 변경 | 하위 호환 |
|---|---|---|
| `model.xml` | `<default>`·`<actuator>` → include 2줄 | 컴파일 결과 동일 (검증) |
| `ik_utils.py` | `DEFAULT_IK_ITERS`, 그리퍼 개도 API 추가 | 기존 3-way 호출 불변 |
| `scene_utils.py` | freejoint 캐시 버그 수정, `TRANSPORT_SMALL_OBJ_REACH` 추가 | 시그니처 불변 |
| `episode_logger.py` | 스키마 레지스트리 (14-D 기본 유지) | `schema` 인자 생략 시 완전 동일 |
| `pi05_ex_infer.py` | `rby1_mobile` obs/action format + 모델 엔트리 | 기존 format 불변 |
| `README.md` | 시나리오·검증 절 추가 | — |

---

## 3. MJCF 구조

```
model.xml (블록, 수정 없음과 동등)      model_transport.xml (신규 기본)
├── defaults_common.xml  ←────공유────→ ├── defaults_common.xml
├── rby1.xml             ←────공유────→ ├── rby1.xml   (수정 없음)
├── scenes/scene_blocks.xml             ├── scenes/scene_transport.xml
└── actuators_arm.xml    ←────공유────→ ├── actuators_arm.xml   (ctrl 0~25)
                                        ├── actuators_base.xml  (ctrl 26~28)
                                        └── <contact><exclude>  ★ 필수
```

### 씬 물체

| 물체 | 충돌 | 시각 | 질량 |
|---|---|---|---|
| 크레이트 | box 13개 (벽·바닥·손잡이 봉/기둥/암) | 23개 (벽·림 프레임·슬랫·손잡이) | 0.815 kg |
| 사과 | 저해상도 convex OBJ 1개 | 5-lobe 사과 + 꼭지 + 잎 OBJ | 0.130 kg |
| 바나나 | 곡률을 따르는 convex OBJ 5개 | curved/tapered/ridged OBJ | 0.084 kg |
| 오렌지 | 저해상도 convex OBJ 1개 | 편평한 구형 껍질 + 별 모양 꽃받침 OBJ | 0.160 kg |
| 배 | 저해상도 convex OBJ 1개 | 좁은 목/넓은 하단 + 꼭지 OBJ | 0.150 kg |
| 선반 | box 6개 (판 3 + 측면 2 + 후면) | 6개 (최상단은 청록색) | 48.96 kg (정적) |
| 사무실 | 벽 4개 (충돌) | 바닥재 + 벽 + 걸레받이 + 유리 파티션 4연창 + 문 | 정적 |

사무실 실내는 x ∈ [-2.9, 3.5], y ∈ [-3.7, 3.1] (**6.4 × 6.8 m**), 벽 높이 2.6 m입니다.
방에 천장이 없으므로 조명 기구(전등 패널)는 두지 않았고, 보이지 않는 `<light>` 4개로만
실내를 밝힙니다.
**바닥은 시각용만 추가**했습니다 — 물리는 여전히 `rby1.xml`의 `floor` 평면(z=0, 마찰 0.05,
condim 1)을 쓰므로 바퀴 접촉과 베이스 exclude가 그대로 유효합니다. 시각 바닥재는 그 위
1 mm에 덮여 기본 체커보드를 가립니다. 벽은 실제 충돌 geom이라 로봇이 방을 벗어날 수 없고,
작업 공간에서 2.3 m 이상 떨어져 있어 시나리오에 간섭하지 않습니다.

크레이트 외곽 0.18×0.30×0.12 m, 내부 0.164×0.284×0.112 m, 손잡이 `|y|=0.200`,
크레이트 원점 기준 z=+0.085. 선반 판 z = 0.35 / 0.70 / **1.00**(목표).

### 과일 4종 visual과 collision mesh

네 물체는 primitive가 아니라 **서로 분리된 OBJ visual/collision 모델**을 사용합니다.
visual mesh는 렌더링 전용(`group=2`, `contype=0`, `conaffinity=0`, `mass=0`)이고,
collision mesh는 기본 뷰에서 숨겨진 물리 전용(`group=3`, `contype=1`, `conaffinity=1`)입니다.
따라서 시각적 세분화 수준이 질량·관성·접촉 계산량을 바꾸지 않습니다. 네 body에는
사과 0.130 kg, 바나나 0.084 kg, 오렌지 0.160 kg, 배 0.150 kg의 `<inertial>`을 명시했습니다.

바나나는 concave visual mesh 하나를 그대로 충돌에 쓰지 않습니다. MuJoCo mesh 충돌은 convex
hull을 사용하므로 그 경우 안쪽 곡률이 사라집니다. 대신 굽은 중심선을 따라 5개의 겹치는 convex
조각을 배치했고, 그리퍼가 누르는 Y 방향 면은 비교적 평평하게 유지해 기존 box proxy의 파지
안정성을 보존했습니다. `success_checks._object_half_extent()`도 이제 첫 geom만 보지 않고 모든
collision mesh vertex의 합집합을 계산합니다.

OBJ는 외부 다운로드 파일이 아니라 이 저장소에서 결정론적으로 생성하는 MIT 자산입니다
(`src/rby1_description/LICENSE`).
형상 파라미터를 바꾼 뒤 다음 명령으로 17개 파일을 다시 만들 수 있습니다.

```bash
python scripts/generate_transport_prop_meshes.py
```

---

## 4. Python 모듈 상세

### `transport_scene.py` — 씬의 단일 진입점

```python
load_layout_config(path) -> dict          # 검증 포함
validate_layout_config(raw) -> dict       # 도달 불가 형상이면 예외
reset_transport_scene(model, data, config, *, rng, randomize, settle_seconds, on_step) -> SceneState
base_handles(model, *, require_actuators=True) -> BaseHandles
build_state_17(model, data, left, right, base) -> np.ndarray   # qpos 기반
build_action_17(model, data, left, right, base) -> np.ndarray  # ctrl 기반
handle_positions / shelf_target_position / body_position / body_rotation
```

`RandomizationSpec`으로 크레이트 xy·yaw·질량, 마찰, 물체 종류·위치, 선반 y, 목표 단을
랜덤화합니다. 질량·마찰은 `model.body_mass` / `body_inertia` / `geom_friction`을 런타임에
수정하며(재컴파일 없음), **비활성 물체는 삭제하지 않고 `(5 + 0.25i, 5, 0.1)`의 개별
parking bay로 치워 둡니다** —
`nq`가 에피소드마다 바뀌면 keyframe과 모든 qpos 인덱스가 깨지기 때문입니다.

`validate_layout_config`가 강제하는 제약은 전부 실측 기반입니다: 크레이트 x ∈ [0.44, 0.56],
손잡이 `|y| ≤ 0.26`, 소형 물체 `|y| ∈ [0.29, 0.34]`, 도킹-선반 여유 > 0.08 m,
목표 단 = 최상단.

### `bimanual_ik.py` — 양팔 IK

**dual-FrameTask 솔버를 만들지 않았습니다.** 좌/우 팔 DoF가 서로소이고 torso·base가
mask에서 빠져 고정되므로, 기존 `solve_kinematic_ik`를 팔마다 한 번씩 호출한 결과가
정확히 합성됩니다(동시 hover 4.2~4.6 mm 실측). task를 늘리면 사후 velocity mask의
projection이 오히려 나빠집니다.

`BiWaypoint`의 목표는 **ndarray 또는 callable**입니다. callable이면 실행 직전에 다시
평가되므로, 파지 전 웨이포인트는 손잡이/물체가 밀려도 추적합니다. 반대로 **파지 후에는
반드시 고정값**을 써야 합니다 — 잡은 뒤에는 손잡이가 팔과 함께 움직여서 "현재 손잡이 위치"
목표는 영원히 도달하지 못합니다.

### `transport_plan.py` — 웨이포인트 빌더

시나리오 2 = 시나리오 1 + 앞단 패킹이므로 크레이트 구간을 여기에 모았습니다.

```
crate_approach_waypoints   hover(standoff 0.10) → descend      [callable 목표]
crate_lift_waypoints       lift → lift_trim (clear_z 절대 높이) [폐루프 보정]
drive_waypoints            base_turn → base_drive
crate_place_waypoints      shelf_hover(+0.09) → shelf_seat
crate_retract_waypoints    개방 후 수직 후퇴
object_pick_waypoints      obj_over → obj_hover → obj_descend → obj_descend_trim
object_into_crate_waypoints obj_lift → obj_carry → obj_lower
```

파지 자세는 **손으로 지정하지 않고** 리셋 시점의 EE 사이트 회전을 캡처합니다
(`capture_grasp_frames`). IK가 현재 손목 자세 근처에서만 수렴하기 때문이며, 도킹 후
다시 캡처하는 것이 베이스 회전 후 배치가 되는 이유입니다.

### `success_checks.py` — 측정 가능한 성공 조건

```python
check_grasp(model, data, arms, target_body, *, require_lift=False) -> GraspResult
object_in_crate(model, data, object_body) -> bool          # 크레이트 로컬 프레임
crate_on_shelf(model, data, *, level) -> ShelfResult       # 선반 로컬 프레임
StopCheck(predicate, hold_steps=8)                          # 8스텝 디바운스
make_transport_stop_check(*, scenario, level, object_body)
```

두 프레임-상대 판정은 물체/선반 **자기 좌표계**에서 계산하므로 크레이트를 들고 기울인
채로도, 선반 포즈를 랜덤화해도 그대로 유효합니다. 디바운스는 기존
`pi05_ex_infer.make_grid_stop_check`와 동일한 8스텝입니다.

---

## 5. 인터페이스

### 카메라·관측 — **변경 없음**
`zed_left`/`wrist_cam_l`/`wrist_cam_r` → `cam_high`/`cam_left_wrist`/`cam_right_wrist`,
224×224 CHW 그대로입니다.

### state/action — 17-D
```
[L arm 0..5, L grip, R arm 0..5, R grip, base_x, base_y, base_yaw]
```
앞 14차원은 기존과 **완전히 동일**합니다(같은 관절 부분집합, 같은 `abs(qpos)/0.045`
그리퍼 정규화). 베이스 3차원은 절대 world 좌표(m, rad)입니다.

`episode_logger`는 스키마 레지스트리로 확장했고 **기본값은 `rby1_14`** 이므로 기존
수집 스크립트는 인자를 넘기지 않아 동작이 불변입니다.

| 스키마 | dim | robot_type |
|---|---|---|
| `rby1_14` (기본) | 14 | `rby1` |
| `rby1_17_mobile` | 17 | `rby1_mobile` |

### 추론
`pi05_ex_infer.py --model rby1_mobile` — obs/action format `rby1_mobile`, 모델은
`model_transport.xml`을 자동 선택. `ARM_DIMS`는 그대로 두어 BJ/IJ/CD 지표를 기존 실험과
비교할 수 있고, 베이스는 `BASE_DIMS = [14,15,16]`로 따로 봅니다.

### 그리퍼 개도 — 임의 값 지정 가능
```python
set_gripper(data, arm, "open")        # 기존 3-way 그대로 (86.4 mm)
set_gripper(data, arm, 0.6)           # 스트로크의 60 %
set_gripper_width(data, arm, 0.030)   # 패드 간격 30 mm
gripper_width_from_qpos(qpos)         # 현재 간격 읽기
```
변환식 `gap(q) = 2|q| − 0.0036`은 실측이며 명령 폭과 실측 폭이 0.0 mm 오차로 일치합니다.
최대 96.4 mm(조인트 한계), `GRIPPER_OPEN`(−0.045) = 86.4 mm.
`BiWaypoint.gripper`도 float를 받고, 시나리오 CLI에 `--grip-open`이 있습니다.

---

## 6. 실측으로 밝혀낸 사실 — 이 씬을 다룰 때 반드시 알아야 할 것

계획 단계의 가정 중 여러 개가 실제와 달랐고, 작업의 대부분이 이걸 찾아내는 데 들어갔습니다.

| # | 사실 | 근거 수치 | 대응 |
|---|---|---|---|
| 1 | **베이스가 바닥에 박혀 있다** | 정상상태 수직항력 **248,943 N** | `<contact><exclude>` world↔base/wheel |
| 2 | **`impratio=10`이 양팔 파지를 가능하게 한다** | 기본값 1에서 5 mm/s 크리프 → 4초 낙하. 10에서 10초간 3.8 mm | `model_transport.xml` `<option>` |
| 3 | **`class="collision"`은 씬 물체에 못 쓴다** | conaffinity=0 ↔ 손가락도 0 → 접촉 계산 자체가 안 됨 | `prop_collision` 신설 |
| 4 | **`GRASP_PAD_OFFSET`은 0.003** | 패드 geom이 EE 사이트 +0.0032/−0.0019. 0.0244로 두면 봉이 패드 아래 가장자리에 걸려 1초 만에 이탈 | 상수 수정 |
| 5 | **리프트는 절대 높이 + 폐루프 트림** | 팔이 하중으로 ~70 mm 처짐. 개루프는 크레이트가 선반 판보다 낮아져 주행 중 충돌 | `clear_z` + `crate_lift_trim` |
| 6 | **경유 고도는 손가락 기준** | 손끝이 패드 중심보다 ~40 mm 아래. 0.08 m 여유는 6 mm 차이로 스침 | `traverse_clearance=0.14` |
| 7 | **소형 물체는 `\|y\| ≥ 0.29`** | 하강 추종오차: y=0.20→71 mm, 0.24→79, 0.28→51, 0.32→9 (팔뚝이 손잡이에 걸림) | 스폰 위치 이동 |
| 8 | **바나나는 visual/충돌 mesh를 분리** | 단일 곡면 충돌은 convex hull이 되고, 둥근 접촉은 패드 사이에서 튕겨 나감 | visual OBJ + 평평한 면을 가진 convex 5분할 |
| 9 | **선반 목표는 최상단만 가능** | 그리퍼+손목이 크레이트 위로 ~0.25 m 돌출 | 하단은 distractor |
| 10 | **시각용 geom은 질량을 가진다** | `prop_visual`에 mass=0 없으면 크레이트 8.46 kg (실제 0.815) | 클래스에 `mass="0"` |
| 11 | **group 3은 뷰어에서 숨겨진다** | 선반이 아예 안 보였음 | 모든 물체에 시각 twin 추가 |
| 11b | **skybox는 모델당 하나뿐** | `rby1.xml`이 이미 선언 → 추가 시 컴파일 오류 | 실내감은 벽으로 처리 |
| 11c | **3인칭 카메라는 방 안에 있어야 함** | 거리 3.0 m면 카메라가 x=-2.09. 방 확장 후 실내가 x ≥ -2.9라 문제없으나, 5 m 이상이면 벽 바깥 | `RECORD_DISTANCE` 3.0 유지 |
| 12 | **모델은 절대경로로 로드** | 상대경로면 `rby1.xml`의 WHEEL geoms 중복 include에서 XML 오류 (mujoco 3.10.0) | 전 스크립트 절대경로 |
| 13 | **바퀴 ctrlrange가 ±3.14였다** | 조인트는 `limited="false"`인데 `inheritrange`가 기본 range를 가져감 → 반 바퀴 이상 못 돌아 최대 ±0.31 m | velocity 액추에이터로 분리 |
| 14 | **`WHEEL_RADIUS`가 40 % 틀렸다** | 0.0602 → 실측 **0.1004** (베이스 0.2836 m / 바퀴 2.824 rad) | 상수 수정 |

### 잘못된 진단 2건 (되돌림)

- **손잡이 봉 11.3° 회전**: 미끄러짐 원인을 "패드-봉 면 정렬 불량"으로 오진하고 봉만
  돌렸습니다. 실제 원인은 #2(`impratio`)였고, 회전은 손잡이가 기둥과 어긋나 보이게만
  했습니다. 제거 후 정지 12초 크리프 7.6 mm로 문제없음을 확인하고 되돌렸습니다.
- **`object_in_crate` 판정식**: "내부에서 여유 5 mm까지 완전히 들어갈 것"을 요구해,
  32 mm 사과가 안쪽 벽에 닿으면(`|x|`=0.050 > 한계 0.045) **원리적으로 만족 불가능**했습니다.
  물체 중심이 크레이트 바닥 위 + 벽 높이 내이면 통과하도록 수정했습니다.
  → **사과 랜덤화 성공률 3/10 → 10/10**

---

## 7. 검증 결과

| 항목 | 방법 | 결과 |
|---|---|---|
| V1 기존 모델 무손상 | git 원본과 컴파일 A/B | **완전 일치** |
| V1 기존 시나리오 | `scenario1_single_arm.py --block red` | SUCCESS |
| V2 모델 불변식 | `transport_scene.py --self-check` | PASSED |
| V3 IK 도달성 | `transport_scene.py --reach-report` | 최대 잔차 **2.0 mm** |
| V4 레이아웃 | `preview_transport_layout.py` | 좌표·여유 확인 |
| V5 시나리오 | 과일 4종 결정적 실행 | 전부 SUCCESS, 크레이트 배치 기울기 0° |
| V6 랜덤 배치 | 100 seeds | 과일 중심 최소 간격 0.080 m 이상, 겹침 없음 |
| V8 데이터셋 | 17-D 수집 + 저장소 검증기 | `[17]` / `rby1_mobile`, 14-D는 `[14]` / `rby1` 불변 |
| V9 추론 배선 | obs/action/stop-check | state (17,), ctrl[26:29] 기록, stop-check 발화 |

---

## 8. 미해결 / 알려진 문제

1. **휠 모드 제자리 회전 불가**
   직진은 완전한 no-slip 구름으로 정상 동작합니다(바퀴 −1.90 rad/s → 베이스 +0.201 m/s).
   회전은 명령 0.758 rad/s 대비 실제 0.13 rad/s이며, 원인은 바퀴 스톨입니다(yaw가 실제
   바퀴 속도의 구름 예측과 정확히 일치 = 미끄러짐 0). 근본 원인은 `base`에 z 자유도가 없어
   접촉 수직력이 바퀴당 **28 kN**(실제 무게의 50배)이 되는 것입니다. 접촉 강성·armature·
   damping을 낮추는 시도는 모두 발산(NaN)했습니다.
   제대로 고치려면 `rby1.xml`의 `base`에 수직 자유도가 필요한데, 이 파일은 블록 파이프라인과
   공유되고 transport 모델 `nq`가 66→67로 바뀌어 기존 keyframe·데이터셋 인덱싱이 깨집니다.
   **선택지**: ① `rby1.xml` 수정 + 블록 keyframe 동시 갱신, ② 휠 전용 로봇 파일 복제,
   ③ 직진만 되는 현 상태 유지. **데이터 수집에는 kinematic 모드(기본)를 쓰세요.**

2. **과일 4종 전체 task randomization sweep 미실행** — 배치 자체는 100개 seed에서
   최소 중심 간격 0.080 m를 확인했지만, 각 과일의 random pose 전체 운반 성공률은 별도 측정이 필요합니다.

3. **17-D의 base 차원이 절대 world 좌표** — 다른 시작 포즈로 일반화되지 않습니다.
   중요하다면 대량 수집 전에 base-frame delta로 바꿔야 하며, 변경은 `build_action_17`
   한 함수지만 이전 데이터셋이 무효화됩니다.

4. **`collect_batch.py`는 기존에 이미 깨져 있음** — `scenario1_single_arm`에서
   `make_waypoints`를 import하는데 실제로는 `make_waypoints_pre_grasp`/`_post_grasp`만
   존재합니다. 이번 작업과 무관한 사전 문제이며 손대지 않았습니다.

---

## 9. 다음 단계

- 17-D 데이터셋 대량 수집 → `openpi`에 `pi05_rby1_mobile_lora` TrainConfig 추가
  (`pi05_ex_infer.MODELS["rby1_mobile"]["checkpoint"]`가 현재 `None`)
- 바나나 실패 시드 원인 분석
- 휠 모드 회전 문제에 대한 방향 결정
