# RBY1 manipulation Python 패키지 구조

## 목적

`rby1_manipulation`의 평면적인 스크립트 모음을 기능별 Python 패키지로 분리한다.
알고리즘·제어·시뮬레이션·데이터 수집의 경계를 명확히 하여 MuJoCo에서 검증한
코드를 이후 `RBY1CommandAdapter` 같은 실물 로봇 어댑터와 연결하기 쉽게 만드는
것이 목적이다.

이 변경은 `rby1_description`의 MJCF/mesh를 수정하지 않는다. `rby1_bringup`에서는
실험 추론기 `pi05_ex_infer.py`의 import 경로만 새 패키지 API로 바꾼다.

## 설치

워크스페이스 루트에서 editable install을 권장한다.

```bash
python -m pip install -e src/rby1_manipulation
```

설치하지 않고 개발할 때는 다음처럼 소스 루트를 지정할 수 있다.

```bash
PYTHONPATH=src/rby1_manipulation/src \
  python -m rby1_manipulation.tasks.block_pick --headless --block red
```

`rby1_description`을 같은 workspace에서 찾을 수 없는 설치 환경에서는 해당 패키지
디렉토리를 명시한다.

```bash
export RBY1_DESCRIPTION_ROOT=/path/to/rby1_description
```

## 패키지 책임

```text
rby1_manipulation
├── control
│   ├── ik.py             관절/액추에이터 핸들, gripper 변환, mink IK
│   ├── single_arm.py     단팔 Cartesian waypoint 실행기
│   ├── bimanual.py       양팔 waypoint와 동시 IK
│   ├── motion.py         hold/ramp/adaptive close 공통 동작
│   └── mobile_base.py    휠 차동구동 변환과 실행
├── simulation
│   ├── common.py         freejoint 배치, 랜덤화, 팔 선택
│   ├── block_scene.py    블록 씬 이름·모델 경로·body pose
│   └── transport_scene.py 운반 씬 config/state/action/reset
├── planning
│   └── transport.py      크레이트·과일 운반 waypoint 생성
├── evaluation
│   ├── block.py          블록 pick-place 성공 판정
│   └── transport.py      grasp/in-crate/on-shelf/stop 판정
├── tasks                 사용자 실행 시나리오만 배치
├── data                  LeRobot 기록·수집 orchestration
├── tools                 viewer·preview·mocap 진단 도구
├── config                설치되는 JSON 기본 설정
└── paths.py              description/config 경로의 단일 진입점
```

의존 방향은 `tasks/data/tools → planning/evaluation → control/simulation`이다.
저수준 모듈은 실행 시나리오를 import하지 않는다. handoff 시나리오도 다른 시나리오
파일에 의존하지 않고 `control.single_arm`, `simulation.block_scene`,
`evaluation.block`의 공통 API를 사용한다.

## 실행 명령

```bash
# 블록 pick-place / handoff
python -m rby1_manipulation.tasks.block_pick --headless --block red
python -m rby1_manipulation.tasks.handoff_left_to_right --headless --block red
python -m rby1_manipulation.tasks.handoff_right_to_left --headless --block red

# 모바일 운반
python -m rby1_manipulation.tasks.transport_crate --headless
python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object apple

# 설정과 장면 검사
python -m rby1_manipulation.simulation.transport_scene --self-check
python -m rby1_manipulation.tools.preview_block_grid --help

# 데이터 수집
python -m rby1_manipulation.data.collect_dataset --output-dir /path/to/dataset
python -m rby1_manipulation.data.collect_batch --root /path/to/dataset
```

동일 명령의 console script도 `pyproject.toml`에 등록되어 있지만, 개발 문서에서는
어느 모듈을 실행하는지 명확한 `python -m` 형식을 기본으로 사용한다.

## 이전 파일명 매핑

| 이전 파일 | 새 모듈 |
|---|---|
| `ik_utils.py` | `rby1_manipulation.control.ik` |
| `bimanual_ik.py` | `rby1_manipulation.control.bimanual` |
| `motion_utils.py` | `rby1_manipulation.control.motion` |
| `wheel_drive.py` | `rby1_manipulation.control.mobile_base` |
| `scene_utils.py` | `rby1_manipulation.simulation.common` |
| `transport_scene.py` | `rby1_manipulation.simulation.transport_scene` |
| `transport_plan.py` | `rby1_manipulation.planning.transport` |
| `success_checks.py` | `rby1_manipulation.evaluation.transport` |
| `scenario1_single_arm.py` | `rby1_manipulation.tasks.block_pick` |
| `scenario2_left_to_right_handoff.py` | `rby1_manipulation.tasks.handoff_left_to_right` |
| `scenario3_right_to_left_handoff.py` | `rby1_manipulation.tasks.handoff_right_to_left` |
| `scenario_transport_crate.py` | `rby1_manipulation.tasks.transport_crate` |
| `scenario_transport_load_and_carry.py` | `rby1_manipulation.tasks.transport_load_and_carry` |
| `episode_logger.py` | `rby1_manipulation.data.episode` |
| `episode_recording.py` | `rby1_manipulation.data.recording` |
| `collect_dataset.py` | `rby1_manipulation.data.collect_dataset` |
| `collect_batch.py` | `rby1_manipulation.data.collect_batch` |
| `preview_block_grid.py` | `rby1_manipulation.tools.preview_block_grid` |
| `preview_transport_layout.py` | `rby1_manipulation.tools.preview_transport_layout` |

루트의 옛 파일을 다시 만드는 호환 shim은 두지 않는다. 저장소 내부 호출부는 새
모듈 경로로 함께 갱신했으며 외부 스크립트는 위 표에 따라 import/실행 경로를
변경해야 한다.

## 설정 파일 위치

JSON은 wheel/editable install 양쪽에서 포함되도록 import 패키지 아래에 둔다.

- `rby1_manipulation/config/grids/block_grid.json`
- `rby1_manipulation/config/grids/block_false_grid.json`
- `rby1_manipulation/config/transport_layout.json`

코드에서는 파일 상대경로를 직접 조립하지 않고 `rby1_manipulation.paths` 또는 각
scene 모듈의 `DEFAULT_*_CONFIG` 상수를 사용한다.

## 변경 후 검증 기준

- 전체 모듈 compile/import
- 모든 CLI의 `--help`
- JSON 유효성 및 package-data 포함 여부
- `model.xml`, `model_transport.xml`, `model_transport_wheels.xml`의
  `nq/nv/nu`, 조인트명, 액추에이터명 비교
- transport scene `--self-check`
- 대표 block/transport 시나리오 headless 실행

리팩터링 전 기준값은 각각 `(52, 49, 26)`, `(66, 61, 29)`, `(66, 61, 26)`이다.
