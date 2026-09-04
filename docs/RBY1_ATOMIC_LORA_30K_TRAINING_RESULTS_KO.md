# RBY1 Atomic Basket π0.5 LoRA 30K 학습 결과

## 1. 결론

2026-08-31에 최신 atomic basket 학습 로그와 checkpoint를 확인한
결과, **30,000 iteration 최적화와 최종 checkpoint 저장은 정상
완료**되었다.

최종 checkpoint는 0부터 시작하는 step 번호를 사용하므로 `29999`로
저장되었다. 로그에서는 최종 asynchronous save가 임시 디렉터리에서
정식 디렉터리로 finalize되었고, background save error가 없으며, main
thread가 finalize 완료를 기다린 뒤 종료한 것이 확인된다.

학습 loss와 gradient norm도 안정적으로 감소했고 NaN, Inf, OOM,
traceback은 발견되지 않았다. 다만 **이 결론은 학습 프로세스와
checkpoint 생성이 완료되었다는 의미**다. 현재 산출물에는 held-out
validation/test loss, MuJoCo closed-loop success rate, 실기 성공률이 없으므로
정책의 작업 성공률까지 검증되었다고 판정할 수는 없다.

| 구분 | 판정 |
|---|---|
| 30K 최적화 | 완료 |
| 최종 checkpoint finalize | 완료 (`29999`) |
| Checkpoint 구조/필수 파일 | 6개 step 모두 확인 |
| 학습 수치 안정성 | 정상, 비정상 수치 없음 |
| Held-out 성능 평가 | 미수행 |
| MuJoCo/실기 rollout 평가 | 미수행 |

---

## 2. 확인 대상

| 항목 | 값 |
|---|---|
| OpenPI config | `pi05_rby1_atomic_lora` |
| Experiment | `rby1_atomic_basket_14d_v2_30k_20260825` |
| Dataset | `data/rby1_atomic_basket_14d_v2` |
| 학습 로그 | `logs/rby1_atomic_lora_30k_20260825.log` |
| 실행 guard 로그 | `logs/rby1_atomic_guard_20260825.log` |
| Checkpoint root | `checkpoints/pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825` |
| 최종 checkpoint | `checkpoints/pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825/29999` |
| 별도 norm stats | `assets/pi05_rby1_atomic_lora/local/rby1_atomic_basket_14d_v2/norm_stats.json` |
| Checkpoint 내 norm stats | `29999/assets/local/rby1_atomic_basket_14d_v2/norm_stats.json` |

`29999` 하위에는 Orbax `_CHECKPOINT_METADATA`, `params`, `train_state`,
`assets` 구성요소가 모두 존재한다. 최종 step의 용량은 약 8.9 GB이며
대략적으로 `params` 6.0 GB, `train_state` 3.0 GB, `assets` 16 KB다.

---

## 3. 학습 설정

`openpi/src/openpi/training/config.py`와 실행 스크립트를 기준으로 한 실제
설정은 다음과 같다.

| 항목 | 설정 |
|---|---|
| Base model | `gs://openpi-assets/checkpoints/pi05_base/params` |
| Fine-tuning | PaliGemma `gemma_2b_lora` + action expert `gemma_300m_lora` |
| 총 iteration | 30,000 |
| Batch size | 32 |
| LR schedule | cosine decay |
| Warmup | 1,000 step |
| Peak LR | `5e-5` |
| Final LR | `5e-6` |
| EMA | 미사용 |
| Log interval | 100 step |
| Save interval | 5,000 step |
| 정규화 | quantile normalization |
| Action 변환 | 그리퍼를 제외한 12-D joint delta action |
| 입력 카메라 | high, left wrist, right wrist 3개, 224×224 |
| State / action | 양팔 14-D |
| Action chunk | 50 step, model action dimension 32로 padding |
| W&B | disabled |

---

## 4. 학습 데이터 범위

전체 1,989 episode 중 연속 구간 `0:1591`만 normalization과 학습에
사용되었다. validation과 test는 config의 `episodes`에 포함되지 않아
held-out 상태를 유지했다.

| Split | Episode 구간 | Episode | Frame |
|---|---|---:|---:|
| Train | `0:1591` | 1,591 | 497,754 |
| Validation | `1591:1790` | 199 | 62,259 |
| Test | `1790:1989` | 199 | 62,248 |
| 전체 | `0:1989` | 1,989 | 622,261 |

데이터셋은 15 FPS이며, 전체 기록 시간은 약 11시간 31분이다. 학습
구간은 약 9시간 13분에 해당한다.

언어 명령은 다음 5개다.

- `put the orange in the basket`
- `put the apple in the basket`
- `put the banana in the basket`
- `put the pear in the basket`
- `lift the basket`

학습 전 preflight에서 다음을 확인했다.

- 1,989개 Parquet과 622,261 row
- train 1,591 episode과 497,754 row
- 3개 카메라의 MP4 5,967개 전체 decode 및 episode 프레임 수 일치
- 15 FPS nominal timestamp grid
- 14-D state/action schema
- 모든 atomic episode의 `success=true`
- 총 199/199 episode의 validation/test 분리

---

## 5. 최적화 결과

로그는 100 step 간격으로 300개 metric record를 남겼다. 최종
checkpoint는 step 29999이지만 metric의 마지막 record는 step 29900이다.
프로세스는 2026-08-25 14:40:35 UTC에 시작해 2026-08-26 02:41:50
UTC에 최종 저장을 마쳤으며, setup과 checkpoint save를 포함한 wall time은
약 12시간 1분이다. 학습 progress bar의 elapsed time은 11시간 57분이다.

| Step | Loss | Gradient norm | Parameter norm |
|---:|---:|---:|---:|
| 0 | 0.1236 | 1.2182 | 1803.8630 |
| 100 | 0.0644 | 0.3639 | 1803.8630 |
| 1,000 | 0.0071 | 0.0457 | 1803.9443 |
| 5,000 | 0.0027 | 0.0236 | 1804.7115 |
| 10,000 | 0.0018 | 0.0202 | 1805.3965 |
| 15,000 | 0.0015 | 0.0197 | 1805.8169 |
| 20,000 | 0.0011 | 0.0175 | 1806.0192 |
| 25,000 | 0.0010 | 0.0168 | 1806.0911 |
| 29,900 | 0.0008 | 0.0151 | 1806.1166 |

100-step 로그 값으로 요약하면 다음과 같다.

- step 0–900 평균 loss: `0.029620`
- step 29,000–29,900 평균 loss: `0.000850`
- 초기 구간 대비 마지막 구간 평균 loss 감소: `97.13%`
- step 25,000–29,900 평균 loss: `0.000912`
- 마지막 1,000 step loss 범위: 로그 표시 정밀도에서 대체로
  `0.0008–0.0009`

이 추세는 발산 없이 최적화가 수렴했음을 보여준다. 다만 이 loss는
학습 batch의 model objective이며, fruit placement 성공률이나 basket lift
성공률을 직접 의미하지 않는다.

---

## 6. Checkpoint 검증

보존된 step은 다음 6개다.

| Step | 약식 용량 | 필수 구성 | 저장 finalize |
|---:|---:|---|---|
| 5,000 | 8.9 GB | 완료 | 완료 |
| 10,000 | 8.9 GB | 완료 | 완료 |
| 15,000 | 8.9 GB | 완료 | 완료 |
| 20,000 | 8.9 GB | 완료 | 완료 |
| 25,000 | 8.9 GB | 완료 | 완료 |
| 29,999 | 8.9 GB | 완료 | 완료 |

전체 보존 용량은 약 54 GB다. 각 step에서 다음 파일/디렉터리가
비어 있지 않은 상태로 확인되었다.

- `_CHECKPOINT_METADATA`
- `params/_METADATA`
- `train_state/_METADATA`
- `assets/local/rby1_atomic_basket_14d_v2/norm_stats.json`

Orbax의 `orbax-checkpoint-tmp` 디렉터리도 남아 있지 않다. 최종
checkpoint의 metadata 기준 저장은 2026-08-26 02:41:25 UTC에 시작해
02:41:50 UTC에 commit되었다.

본 확인은 디스크 상의 구조, metadata, 필수 asset, 저장 finalize를
검증한 것이다. 전체 tensor를 다시 deserialize하는 추론 smoke test는
아직 수행하지 않았다.

---

## 7. 로그 경고 해석

학습 완료 판정에 영향을 주지 않는 경고는 다음과 같다.

1. JAX가 사용하지 않는 ROCm과 TPU backend 초기화에 실패했다.
   이후 data loader, base checkpoint restore, 30K update, 모든 checkpoint
   save가 정상 진행되었으므로 실행 실패로 보지 않는다.
2. LeRobot v2.0의 global stats 포맷을 사용했다는 하위 호환성
   경고가 있다. 현재 버전에서는 backward-compatible로 load되었지만,
   다음 학습 전에는 데이터셋 포맷 업그레이드를 검토한다.
3. Single-process 학습이어서 Orbax cross-host metadata validation이
   skip되었다. 이는 error가 아니다.
4. W&B가 disabled였으므로 별도 dashboard metric은 없다. 본 문서의
   수치는 local text log에서 추출했다.

Guard 로그에 남은 이전 두 번의 `norm computation did not report
successful output`은 정규화 통계 생성을 기다리던 사전 실행의
중단 기록이다. 최종 학습은 norm stats와 전체 preflight를 통과한
뒤 2026-08-25 14:40 UTC에 별도로 시작했다.

---

## 8. 평가 권장 절차

현재 상태에서는 `29999`를 **최종 학습 candidate**로 사용할 수
있지만, 최종 배포 checkpoint로 확정하기 전에 다음 순서로 평가한다.

1. `29999` checkpoint를 params와 동일 checkpoint 내 `norm_stats.json`으로
   로드하는 inference smoke test를 수행한다.
2. Validation 199 episode에서 `20000`, `25000`, `29999` checkpoint를
   비교한다. Train loss만으로는 최적 checkpoint를 선택할 수 없다.
3. Checkpoint 선택과 threshold 조정을 완료한 후 test 199 episode를
   단 한 번 평가한다.
4. MuJoCo closed-loop에서 네 과일 넣기와 basket lift를 명령별,
   scenario family별로 집계한다.
5. 실기 적용 전에 action range, joint limit, gripper 표현, 안전 정지,
   camera mapping을 다시 확인한다.

보고해야 할 최소 metric은 명령별 episode 수, 성공 수/성공률,
wrong-object grasp, drop, 미완료, timeout, joint-limit/safety-stop 수다. 이 평가가
추가되기 전까지는 본 실험의 상태를 **training complete, policy quality
pending**으로 표기한다.

---

## 9. 재확인 명령

학습 metric은 다음과 같이 확인할 수 있다.

```bash
cd /mnt/dev/work/pi05_TO_hybrid
tr '\r' '\n' < logs/rby1_atomic_lora_30k_20260825.log \
  | rg 'Step [0-9]+: grad_norm='
```

보존 checkpoint와 용량은 다음과 같이 확인한다.

```bash
find checkpoints/pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825 \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n
du -sh checkpoints/pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825/*
```

최종 저장 finalize 기록은 다음으로 확인한다.

```bash
rg 'Saving checkpoint at step 29999|Finished saving checkpoint|Done waiting' \
  logs/rby1_atomic_lora_30k_20260825.log | tail -n 5
```
