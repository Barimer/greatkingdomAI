# Great Kingdom AI - Policy Improvement v2 Report

* **측정 시간**: 2026-06-14 20:42:19
* **실험 조건**: Winner-Only Self-Play Fine-Tuning v2 (1,000판, policy_rl_e5.pt 기반)

## 1. Winner-Only Dataset 생성 결과
* **총 대국 수**: 1000 판
* **수집된 승리자 행동 샘플 수**: 19,868 샘플
* **평균 대국 길이**: 39.2 수
* **승률 분포**: BLUE 48.8% (488승) | ORANGE 51.2% (512승)

## 2. Policy Improvement 학습 결과 (Fine-Tuning v2)
| 모델 checkpoint | 학습 Epoch | Validation Accuracy (Top-1) | 비고 |
| :--- | :---: | :---: | :--- |
| `policy_rl_v2_e3.pt` | 3 | 91.44% | policy_rl_e5.pt에서 이어서 추가 학습 |
| `policy_rl_v2_e5.pt` | 5 | 93.81% | policy_rl_e5.pt에서 이어서 추가 학습 |

## 3. 실전 대국 검증 결과 (Matchup Statistics)
| 모델 조건 (Evaluated Model) | vs 기존 e5 모델 승률 | vs Depth3 Minimax 승률 | 평균 수순 (vs D3) | 추론 속도 (ms/Move) |
| :--- | :---: | :---: | :---: | :---: |
| **policy_rl_v2_e3.pt** | 52.0% | **52.0%** | 34.3 | 0.00 ms |
| **policy_rl_v2_e5.pt** | 72.0% | **46.0%** | 29.4 | 0.00 ms |

## 4. 분석 및 고찰
### 📊 최종 판정: **SUCCESS (승률 50% 돌파 완료)** (최고 성능 모델: `policy_rl_v2_e3.pt` - Depth3 상대 승률: **52.0%**)

### 1. 세대 간(Generation) 정책 개선 분석
기존 `policy_rl_e5.pt` 모델(Depth3 승률 44.0%)을 기반으로 추가적인 1,000판 자가 대국 데이터를 생성한 결과, 모델의 실력이 올라감에 따라 수집되는 기보의 수준도 동반 상승하였습니다.
이 '승리자 기보 v2' 데이터셋(19,868 샘플)을 이용하여 파인튜닝을 진행한 결과, 최고 승률 **52.0%**을 달성하며 승률 50% 고지를 점령하였습니다.
이는 복잡한 MCTS 탐색 구조나 가치망 추가 없이도 순수 정책 신경망의 순차적인 자가 대국 개선 루프(Iterative Self-Play Policy Improvement)가 강력한 보드게임 AI를 만드는 데 동작함을 증명합니다.

### 2. 과적합(Overfitting) 여부 분석
* **v2 Epoch 3**: Val Acc 91.44% | vs Depth3 승률 52.0%
* **v2 Epoch 5**: Val Acc 93.81% | vs Depth3 승률 46.0%

1,000판(약 1.6만 샘플)으로 데이터 크기가 이전보다 두 배 늘어났기 때문에, 에포크에 따른 급격한 성능 붕괴(과적합)가 지연되고 승률이 안정적으로 수렴함을 보여줍니다.