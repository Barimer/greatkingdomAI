# Great Kingdom AI - Policy RL v3 Training Report

* **측정 시간**: 2026-06-16 00:14:46
* **실험 조건**: Pure Self-Play RL Fine-Tuning v3 (8,000판, policy_rl_v2_e3.pt 기반)

## 1. 자가 대국(Self-Play) 생성 통계
* **총 대국 수**: 8000 판
* **수집된 승리자 행동 샘플 수**: 148,074 샘플
* **평균 대국 길이**: 36.7 수
* **승률 분포**: BLUE 35.0% (2797승) | ORANGE 65.0% (5203승)

## 2. Policy RL v3 학습 결과 (Fine-Tuning v3)
| 모델 checkpoint | 학습 Epoch | Validation Loss | Validation Accuracy (Top-1) | vs RL v2 (200판) |
| :--- | :---: | :---: | :---: | :---: |
| `policy_rl_v3_e1.pt` | 1 | 0.2254 | 95.01% | 69.5% |
| `policy_rl_v3_e2.pt` | 2 | 0.1556 | 95.86% | 71.5% (최우수) |
| `policy_rl_v3_e3.pt` | 3 | 0.1370 | 96.10% | 67.0% |
| `policy_rl_v3_e4.pt` | 4 | 0.1352 | 95.68% | 70.5% |
| `policy_rl_v3_e5.pt` | 5 | 0.1334 | 95.51% | 62.5% |

## 3. 최우수 모델 실전 대국 검증 결과 (vs Depth3 Minimax)
| 모델 조건 (Evaluated Model) | vs 기존 RL v2_e3 모델 (200판) | vs Depth3 Minimax (80판) | 평균 수순 (vs D3) |
| :--- | :---: | :---: | :---: |
| **policy_rl_v3_e2.pt** | 71.5% | **35.0%** | 32.2 |

## 4. 최종 판정 및 고찰
### 📊 최종 결과: **FAIL (Champion Kept)**
- **챔피언 교체 조건**: vs RL v2 >= 55% 및 vs Depth3 >= 52%
- **실제 성능**: vs RL v2 **71.5%** (만족), vs Depth3 **35.0%** (불만족)

### 1. 세대 간(Generation) 정책 개선 분석 및 성능 향상 검증
이전 세대의 최우수 모델인 `policy_rl_v2_e3.pt` (Depth3 승률 52.0%)를 기반으로 8,000판의 자가 대국을 진행하여 승리자 기보 데이터셋(148,074 샘플)을 구축하였습니다.
기존 v2 대비 fine-tuning learning rate를 0.2배 수준으로 대폭 축소하여 학습이 급격하게 기존 가치를 덮어쓰지 않고 점진적으로 최적화되도록 설계하였습니다.
그 결과 최우수 모델인 `policy_rl_v3_e2.pt`이 RL v2 모델을 상대로 **71.5%**의 압도적인 승률을 보여주며 확실한 세대 개선을 증명하였습니다.
특히, 외부 규칙 기반 검증 모델인 Depth3 Minimax를 상대로는 **35.0%**의 승률을 기록하여, 성공 기준선인 52%를 돌파하지 못하고 챔피언 교체에 실패하였습니다.
Depth3 Minimax 상대 승률이 기준인 52%에 도달하지 못하여 기존 챔피언 `policy_rl_v2_e3.pt`를 유지합니다.

### 2. 향후 추가 개선 방향
- **자가 대국 편향 극복**: Winner-Only Behavioral Cloning(행동 복제) 학습은 생성된 자가 대국 데이터 내의 시나리오만을 모방하게 됩니다. 상대방이 변칙수를 두거나 Depth3 미니맥스처럼 정교한 탐색 공격을 펼칠 때 대응 능력이 저하되는 현상이 발생하기 쉽습니다.
- **MCTS 및 Value Network 융합**: 차세대 모델에서는 순수 정책 신경망을 넘어 가치망(Value Network)과 몬테카를로 트리 탐색(MCTS)을 결합하여, 단순 모방 학습을 극복하고 탐색을 통한 기력 향상을 노려야 합니다.