# Great Kingdom AI - Dataset Integrity Verification Report

* **측정 시간**: 2026-06-13 19:31:30
* **검증 대상**: `data/selfplay_dataset_v1.npz`

## 1. Dataset Keys & Types
| Key | Shape | Dtype |
| :--- | :---: | :---: |
| `states` | (41728, 4, 9, 9) | `int8` |
| `actions` | (41728, 2) | `int8` |
| `players` | (41728,) | `int8` |
| `results` | (41728,) | `int8` |
| `game_ids` | (41728,) | `int32` |
| `terminations` | (1000,) | `<U10` |

## 2. Memory Usage & Compression
* **states**: 12.8936 MB
* **actions**: 0.0796 MB
* **players**: 0.0398 MB
* **results**: 0.0398 MB
* **game_ids**: 0.1592 MB
* **terminations**: 0.0381 MB
* **총 예상 uncompressed 용량**: **13.2501 MB**
* **실제 npz 저장 용량**: **0.5295 MB**
* **압축률 (Compression Ratio)**: **25.02x** (압축으로 인해 용량이 작게 나타난 것임)

## 3. Tensor Content Validation
* **각 좌표 채널 합이 1인지 여부 (One-hot Grid Validation)**: **True**
* **셀 통계 (Cell Occupancy)**:
  * 내 돌 (Channel 0): 495417개 (14.66%)
  * 상대 돌 (Channel 1): 515794개 (15.26%)
  * 중립 성 (Channel 2): 41728개 (1.23%)
  * 빈 칸 (Channel 3): 2327029개 (68.85%)
  * 총 합 검증: 3379968 / 3379968 개 (성공)

## 4. Action Validation
* **Action 좌표 범위**: `[-1, 8]` (합법 범위 내 존재)
* **착수 구분**:
  * 바둑판 착수 (Board Moves): 41186회 (98.7%)
  * 패스 (Pass): 542회 (1.3%)
* **바둑판 착수 9x9 히스토그램 (Action Grid Distribution)**:
```
Row 0: 149 454 472 453 418 439 462 482 154
Row 1: 459 767 660 652 583 635 656 757 485
Row 2: 482 651 535 483 463 465 526 639 491
Row 3: 437 596 506 465 217 462 518 651 453
Row 4: 453 623 492 225   0 228 488 642 485
Row 5: 480 662 509 492 238 522 516 657 467
Row 6: 495 669 550 530 520 536 556 679 495
Row 7: 519 805 673 671 663 664 654 807 510
Row 8: 174 505 519 467 472 487 519 523 168
```

## 5. Result Validation
* **결과값 종류**: [1, 2]
* **BLUE 승리 샘플 수**: 18095개 (43.4%)
* **ORANGE 승리 샘플 수**: 23633개 (56.6%)

## 6. First Sample Dump
### states[0] (Channels: My, Opp, Neutral, Empty)
```
Channel 0:
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
Channel 1:
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
Channel 2:
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
Channel 3:
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 0 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1
```
* **actions[0]**: [0, 8]
* **players[0]**: 1 (BLUE)
* **results[0]**: 2 (ORANGE)

## 7. Random 5 Samples Dump
### Sample Index: 15837
* **action**: [2, 5]
* **player**: 1 (BLUE)
* **result (winner)**: 2 (ORANGE)
* **Stones Count**: My=19, Opp=19, Neutral=1, Empty=42
### Sample Index: 7836
* **action**: [3, 5]
* **player**: 1 (BLUE)
* **result (winner)**: 1 (BLUE)
* **Stones Count**: My=22, Opp=22, Neutral=1, Empty=36
### Sample Index: 21610
* **action**: [6, 6]
* **player**: 1 (BLUE)
* **result (winner)**: 1 (BLUE)
* **Stones Count**: My=11, Opp=11, Neutral=1, Empty=58
### Sample Index: 31681
* **action**: [0, 5]
* **player**: 2 (ORANGE)
* **result (winner)**: 1 (BLUE)
* **Stones Count**: My=6, Opp=7, Neutral=1, Empty=67
### Sample Index: 8288
* **action**: [4, 1]
* **player**: 1 (BLUE)
* **result (winner)**: 2 (ORANGE)
* **Stones Count**: My=14, Opp=14, Neutral=1, Empty=52

## 8. 최종 판정 (Final Judgment)
### 판정 결과: **PASS**

### 상세 사유:
1. All checks passed successfully. The dataset is fully valid and ready for training.