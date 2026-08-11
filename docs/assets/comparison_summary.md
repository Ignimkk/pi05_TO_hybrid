# Baseline vs SEAM — RBY1 grid evaluation

- paired trials: **216** (baseline 216, seam 216, unmatched 0)
- grid revision: `006698dac3ef`

## 1. Task success

| condition | success | rate | 95% CI (Wilson) |
|---|---:|---:|---|
| baseline | 170/216 | 78.7% | [72.8%, 83.6%] |
| SEAM | 173/216 | 80.1% | [74.3%, 84.9%] |

Paired outcomes (same grid cell, same repeat index):

| | SEAM success | SEAM fail |
|---|---:|---:|
| **baseline success** | 163 | 7 |
| **baseline fail** | 10 | 36 |

McNemar exact test on the 17 discordant pairs: **p = 0.629**

## 2. Success by workspace zone

| zone | n | baseline | SEAM | delta |
|---|---:|---:|---:|---:|
| near (x < 0.625) | 162 | 160/162 (98.8%) | 162/162 (100.0%) | +1.2 pp |
| far  (x = 0.625) | 54 | 10/54 (18.5%) | 11/54 (20.4%) | +1.9 pp |

## 3. Motion quality (recomputed from trajectory NPZs, 12 arm joints)

Trials with a usable NPZ in both conditions: **216** (skipped 0)

### action space (commanded joint targets)

| metric | baseline (mean ± sd) | SEAM (mean ± sd) | change | Wilcoxon p |
|---|---:|---:|---:|---:|
| BJ | 0.02802 ± 0.00407 | 0.02147 ± 0.00366 | -23.4% | <0.001 |
| IJ | 0.01274 ± 0.00108 | 0.00919 ± 0.00088 | -27.9% | <0.001 |
| CD | 0.02289 ± 0.00396 | 0.01810 ± 0.00290 | -20.9% | <0.001 |
| AVb | 0.00020 ± 0.00008 | 0.00014 ± 0.00006 | -31.7% | <0.001 |

### qpos space (measured physical response)

| metric | baseline (mean ± sd) | SEAM (mean ± sd) | change | Wilcoxon p |
|---|---:|---:|---:|---:|
| BJ | 0.00362 ± 0.00062 | 0.00296 ± 0.00048 | -18.1% | <0.001 |
| IJ | 0.00318 ± 0.00041 | 0.00267 ± 0.00038 | -15.9% | <0.001 |
| CD | 0.01381 ± 0.00332 | 0.01551 ± 0.00273 | +12.4% | <0.001 |
| AVb | 0.00001 ± 0.00000 | 0.00000 ± 0.00000 | -25.9% | <0.001 |

### Overlap residual (previous chunk's tail vs. new chunk's head)

| baseline | SEAM | change | Wilcoxon p |
|---:|---:|---:|---:|
| 0.07549 | 0.06355 | -15.8% | <0.001 |

## 4. Inference cost

| condition | mean latency (ms) | max (ms) | chunks/trial |
|---|---:|---:|---:|
| baseline | 218.6 | 6201.9 | 34.8 |
| SEAM | 219.5 | 12893.0 | 32.8 |

