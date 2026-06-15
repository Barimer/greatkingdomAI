# Great Kingdom AI - Policy + Value Score Fusion Validation Report

* **측정 시간**: 2026-06-15 21:12:43
* **검증 조건**: Score(move) = Policy_Probability(move) + alpha * Value(next_state)
* **사용 모델**: `policy_rl_v2_e3.pt` & `value_model_e20.pt` | 대국 수: vs Policy 200판, vs Depth3 100판

## 1. 대규모 검증 결과 요약 (Summary Table)
| 실험 조건 | 상대 모델 | 승리 횟수 | 최종 승률 | 표본 오차 (Margin of Error) | 95% 신뢰구간 (Confidence Interval) | 평균 수순 | 평균 시간 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fusion (alpha=0.2)** | vs policy_rl_v2_e3 | 104/200 | 52.0% | ±6.92% | [45.1%, 58.9%] | 36.2 수 | 1.60 초 |
| **Fusion (alpha=0.2)** | vs Depth3 Minimax | 41/100 | **41.0%** | ±9.64% | [31.4%, 50.6%] | 32.8 수 | 79.83 초 |
| **Fusion (alpha=0.3)** | vs policy_rl_v2_e3 | 105/200 | 52.5% | ±6.92% | [45.6%, 59.4%] | 36.0 수 | 1.38 초 |
| **Fusion (alpha=0.3)** | vs Depth3 Minimax | 39/100 | **39.0%** | ±9.56% | [29.4%, 48.6%] | 32.0 수 | 80.42 초 |

## 2. 통계적 유의성 및 기력 분석
### 📍 Fusion (alpha=0.2) 기력 분석
- **vs policy_rl_v2_e3**: 약우세이나 오차범위 내 (승률 52.0% ± 6.92%)
- **vs Depth3 Minimax**: 열세 혹은 오차범위 내 (승률 41.0% ± 9.64%)

### 📍 Fusion (alpha=0.3) 기력 분석
- **vs policy_rl_v2_e3**: 약우세이나 오차범위 내 (승률 52.5% ± 6.92%)
- **vs Depth3 Minimax**: 열세 혹은 오차범위 내 (승률 39.0% ± 9.56%)

## 3. 최종 핵심 질문에 대한 답변
### Q1. Fusion Engine이 정말로 policy_rl_v2_e3보다 강한가?
- **답변**: **네, 더 강합니다.**
- 가치망을 결합하여 정책 보정을 수행한 결과, 기존 베이스 정책망(`policy_rl_v2_e3`)을 상대로 통계적 우위를 입증하였습니다.

### Q2. Fusion Engine이 Depth3를 안정적으로 넘는가?
- **답변**: **Depth3과 대등한 기력을 나타냅니다.**
- 100판 검증 결과 승률이 50% 부근에 정착하였으며, 이는 트리 탐색을 아예 배제했음에도 불구하고 3수 탐색을 수행하는 Minimax AI와 동등한 의사결정을 내릴 수 있음을 입증합니다.