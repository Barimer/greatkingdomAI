# Great Kingdom AI - Hybrid RL Experiment Report

* **측정 시간**: 2026-06-14 23:45:00
* **실험 조건**: Hybrid RL Fine-Tuning (80% Winner-Only Self-Play + 20% Fast Depth3 Teacher, policy_rl_v2_e3.pt 기반)

## 1. Dataset 생성 및 혼합 결과
* **총 Self-Play 대국 수**: 1000 판
* **수집된 RL 승리자 샘플 수 (80%)**: 18,858 샘플
* **주입된 Depth3 Teacher 샘플 수 (20%)**: 4,714 샘플
* **총 혼합 학습 샘플 수**: 23,572 샘플
* **자가 대국 평균 길이**: 37.4 수
* **자가 대국 승률 분포**: BLUE 33.0% (330승) | ORANGE 67.0% (670승)

## 2. Policy Improvement 학습 결과 (Fine-Tuning)
| 모델 checkpoint | 학습 Epoch | Validation Accuracy (Top-1) | 비고 |
| :--- | :---: | :---: | :--- |
| `policy_hybrid_rl_e2.pt` | 2 | 81.09% | policy_rl_v2_e3.pt에서 이어서 추가 학습 |
| `policy_hybrid_rl_e3.pt` | 3 | 83.50% | policy_rl_v2_e3.pt에서 이어서 추가 학습 |

## 3. 실전 대국 검증 결과 (Matchup Statistics)
| 모델 조건 (Evaluated Model) | vs 기존 v2_e3 모델 (100판) | vs Depth3 Minimax (100판) | vs Hybrid (100판) | 평균 수순 (vs D3) |
| :--- | :---: | :---: | :---: | :---: |
| **policy_hybrid_rl_e2.pt** | 52.0% | **35.0%** | 36.0% | 35.0 |
| **policy_hybrid_rl_e3.pt** | 63.0% | **36.0%** | 54.0% | 31.4 |

## 4. 분석 및 고찰
### 📊 최종 판정: **FAIL (성공 기준 미달)** (최고 성능 모델: `policy_hybrid_rl_e3.pt` - Depth3 상대 승률: **36.0%**)

### 1. 외부 Teacher 신호 주입을 통한 Self-Play Collapse 방지 실패 분석
이전의 순수 자가 대국 v3 실험(성공 실패, 최고 승률 50%)에서는 학습을 진행할수록 특정 자가 대국 유형에 편향되는 **Self-Play Collapse**와 급격한 기력 저하(36% 수준으로 추락)가 발생하였습니다.
이를 해결하기 위해 본 Hybrid RL 실험에서는 규칙 기반 교사인 **Fast Depth3 Teacher 기보 20%**를 혼합 주입하였으나, 검증 결과 Epoch 2에서 **35.0%**, Epoch 3에서 **36.0%**의 승률을 기록하며 **기존 베이스 모델(52.0%) 대비 성능이 오히려 대폭 하락(Collapse 발생)**하였습니다.

### 2. 성능 급감의 원인 분석
* **약한 교사(Weak Teacher)의 한계**: 주입된 Fast Depth3 데이터셋(`selfplay_fast_depth3_500.npz`)의 실제 성능은 Depth3 Minimax 대비 승률 30% 수준에 불과한 약한 모델입니다. 이와 같이 질적으로 낮은 교사의 행동 양식을 정책망에 강제로 모조(Behavior Cloning)하게 함으로써, 기존 베이스 모델이 갖추고 있던 정교한 수읽기와 포석 감각을 훼손하는 부정적 규제(Negative Regularization)로 작용하였습니다.
* **오프라인 데이터 혼합의 한계**: 단순 오프라인 데이터 혼합과 지도 학습 파인튜닝만으로는 신경망이 자가 대국에서 고착화된 편향(Self-Play Bias)을 스스로 깨고 Depth3의 정밀한 탐색 공격을 막아낼 만한 강인성을 획득하지 못했습니다.

## 5. 최종 핵심 질문에 대한 답변
### Q. 외부 Teacher 신호를 주기적으로 주입하면 Self-Play Collapse를 방지할 수 있는가?
- **답변**: **아닙니다. 단순히 저품질/약한 교사(Weak Teacher)의 신호를 혼합하는 방식으로는 Self-Play Collapse를 방지할 수 없으며, 오히려 기존 기력까지 동반 붕괴(Collapse)시킵니다.**
- 본 실험을 통해 검증되었듯이, 52%의 기력을 가졌던 베이스 모델에 30% 승률 수준의 Fast Depth3 데이터를 섞어 학습하자마자 기력이 35~36%로 즉각 추락했습니다.
- 따라서 Self-Play Collapse를 방지하고 55% 장벽을 돌파하기 위해서는:
  1. 기존 모델보다 명백히 강력한 **강한 교사(Strong Teacher, 예: Depth 4~5 Minimax 또는 고성능 MCTS)**의 기보를 주입해야 합니다.
  2. 단순 Behavior Cloning 방식의 오프라인 학습 한계를 극복하기 위해, **가치 신경망(Value Network)**을 도입하여 형세를 평가하거나 **MCTS 탐색**을 정책 학습 루프에 직접 결합(AlphaZero 스타일)하는 온라인 강화학습 구조로 전환해야 합니다.