# pi05_TO_hybrid 컨테이너 세팅 가이드

로컬 PC에서 수집한 데이터셋과 코드를 사내 GPU 서버 컨테이너로 옮기고
openpi π0.5 LoRA 파인튜닝을 수행하기 위한 절차입니다.

- 로컬 경로: `/home/mk/dev_ws/vla/pi0_TO_ws`
- 컨테이너 경로: `/root/work/pi05_TO_hybrid`
- 데이터셋: `data/rby1_dataset_v1` (1200 에피소드, 428k 프레임, 6 태스크)

---

## Step 3.1: 로컬 → 컨테이너 전송

### 프로젝트 구조 정리 (전송 전, 로컬)

전송할 것:
- `src/rby1_manipulation/` (시나리오 + 데이터 수집 스크립트, 인퍼런스 harness 용)
- `src/rby1_description/`, `src/rby1_bringup/` (MuJoCo 모델 — 인퍼런스 평가 시 필요)
- `data/rby1_dataset_v1/` (학습 데이터)
- `scripts/` (compute_stats, validate_dataset)

전송하지 말 것 (용량 큼, 컨테이너에서 다시 설치):
- `venv/`, `.venv/`, `__pycache__/`, `pi0_TO_env/`
- `data/rby1_dataset_smoke/` (검증용 소량 데이터)

### rsync 전송 (권장, 재개 지원)

로컬 셸에서:
```bash
# 데이터셋만 먼저 전송 (대용량, ~수 GB)
rsync -avP --exclude '__pycache__' --exclude '.venv' \
    /home/mk/dev_ws/vla/pi0_TO_ws/data/rby1_dataset_v1 \
    <container-user>@<container-host>:/root/work/pi05_TO_hybrid/data/

# 코드 (가벼움)
rsync -avP \
    --exclude '__pycache__' --exclude '.venv' \
    --exclude 'data' --exclude 'venv' \
    /home/mk/dev_ws/vla/pi0_TO_ws/ \
    <container-user>@<container-host>:/root/work/pi05_TO_hybrid/
```

SSH 통해 컨테이너 안에 있다면 로컬 SCP:
```bash
# 컨테이너 쪽에서 로컬로 다운로드
scp -r <local-user>@<local-host>:/home/mk/dev_ws/vla/pi0_TO_ws/data/rby1_dataset_v1 \
    /root/work/pi05_TO_hybrid/data/
```

### 컨테이너에서 크기 검증

```bash
cd /root/work/pi05_TO_hybrid
du -sh data/rby1_dataset_v1
find data/rby1_dataset_v1/data/chunk-000 -name "episode_*.parquet" | wc -l  # 1200 나와야 함
```

---

## Step 3.2: openpi 설치

openpi는 Physical Intelligence의 공식 pi0/pi0.5 구현입니다.
- 저장소: https://github.com/Physical-Intelligence/openpi
- JAX 기반 (H200에서 xla/cuda-12 필요)

컨테이너 안에서:

```bash
cd /root/work
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi

# uv (권장) 또는 pip 사용
pip install uv
uv venv --python 3.11
source .venv/bin/activate

# 개발 모드 설치 (LoRA 예제 코드 접근 가능)
uv pip install -e ".[dev]"

# JAX GPU 확인 (CUDA 12)
python -c "import jax; print(jax.devices())"
# 예상 출력: [CudaDevice(id=0)]
```

**H100/H200 사용 시** JAX가 자동으로 flash-attention을 켭니다 (bf16). 별도 설정 불필요.

---

## Step 3.3: 데이터셋을 openpi가 인식하도록 등록

openpi는 LeRobot 포맷을 직접 지원합니다 (`openpi.training.data_loader.LeRobotDataConfig`).

우리 데이터셋 경로를 `LEROBOT_HOME` 환경변수로 지정하거나 명시적 경로 전달:

```bash
export LEROBOT_HOME=/root/work/pi05_TO_hybrid/data
# 이제 openpi가 rby1_dataset_v1을 "rby1_dataset_v1" repo_id로 인식
```

또는 openpi config 내부에서 절대 경로 지정 (아래 config 파일 참고).

---

## Step 3.4: LoRA 파인튜닝 config

openpi 저장소 안 `src/openpi/training/config.py`에 새 config 추가:

**예시 (openpi 최신 버전 기준, API 변경 있을 수 있음):**

```python
# src/openpi/training/config.py 안에 추가

TrainConfig(
    name="pi05_rby1_lora",
    model=pi0_config.Pi0Config(
        # pi0.5 체크포인트 (Physical Intelligence 공개)
        # 또는 pi05_base 등 최신 이름 사용
    ),
    data=LeRobotDataConfig(
        repo_id="rby1_dataset_v1",
        # 우리 관측 스키마 매핑 (14-dim state, 14-dim action, 3 cameras)
        prompt_from_task=True,  # tasks.jsonl의 task 문자열을 prompt로 사용
    ),
    # LoRA 설정
    lora=LoRAConfig(
        rank=32,
        alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ),
    # 학습 하이퍼파라미터 (H200 141GB 기준)
    batch_size=32,
    num_train_steps=30000,        # ~30k steps ≈ 5~10 epochs on our data
    lr_schedule=cosine(warmup=1000, peak_lr=5e-5, decay_steps=30000),
    weight_decay=1e-4,
    checkpoint_dir="/root/work/pi05_TO_hybrid/checkpoints/rby1_lora",
    save_every=5000,
    log_every=100,
),
```

openpi의 exact API는 릴리즈에 따라 다르므로, `openpi/examples/` 아래
LoRA 예제 (e.g., `pi05_droid_finetune.py`, `pi05_libero_finetune.py`)를
참고해 우리 데이터셋용으로 복사·수정하는 것이 안전합니다:

```bash
cp openpi/examples/pi05_libero_finetune.py \
   openpi/examples/pi05_rby1_finetune.py
# repo_id, checkpoint 경로, state/action dim만 우리 값으로 수정
```

---

## Step 3.5: 학습 실행

```bash
cd /root/work/openpi
source .venv/bin/activate
export LEROBOT_HOME=/root/work/pi05_TO_hybrid/data

# 단일 GPU (H200 1장)
python scripts/train.py --config pi05_rby1_lora

# 다중 GPU (있는 경우, openpi는 jax.pmap 지원)
XLA_FLAGS="--xla_force_host_platform_device_count=1" \
    python scripts/train.py --config pi05_rby1_lora
```

**예상 학습 시간 (H200 1장):**
- 30k steps × batch 32
- 프레임 428k / batch 32 = 13.4k step = 1 epoch
- 30k steps ≈ 2.2 epochs
- ~4~6시간

**중간 체크포인트 확인:**
```bash
watch -n 30 'ls -la /root/work/pi05_TO_hybrid/checkpoints/rby1_lora/'
```

---

## Step 4: MuJoCo 인퍼런스 평가 (파인튜닝 후)

학습 완료 후 체크포인트를 로드해 MuJoCo 환경에서 policy inference로 시나리오 실행:

**필요한 파일 (Step 4에서 만들 예정):**
- `src/rby1_manipulation/inference_policy.py`:
  - openpi 체크포인트 로드
  - MuJoCo `data` → observation dict 변환
  - policy(obs, prompt) → action[chunk_size, 14]
  - action chunk를 `data.ctrl`에 순차 적용
  - `check_success()`로 성공률 측정

Step 3까지 완료되면 인퍼런스 harness 작성 지원드리겠습니다.

---

## 트러블슈팅

**JAX가 CUDA 못 찾음:**
```bash
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**LeRobot 데이터 로드 실패:**
- `meta/info.json`의 `codebase_version`이 최신 LeRobot 요구와 다를 수 있음
- openpi가 요구하는 필드가 stats.json에 있는지 확인 (min/max/mean/std)

**OOM (out of memory):**
- `batch_size`를 32 → 16 → 8로 낮춤
- gradient accumulation 사용
- FSDP/sharding 활성화 (openpi 지원)

**LoRA 학습 발산:**
- `peak_lr`을 5e-5 → 1e-5로 낮춤
- `warmup`을 1000 → 3000으로 늘림
