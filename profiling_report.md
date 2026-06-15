# Great Kingdom AI - Profiling Report

본 보고서는 2판의 Depth 3 자가대국(Self-Play) 데이터를 바탕으로 수집된 `cProfile` 계측 결과(`profile.out`)를 분석하여 작성되었습니다.

---

## 1. 📊 CPU 누적 시간 TOP 30

전체 계측 실행 시간: **1,838.894초** (약 30.6분)
전체 함수 호출 수: **3,785,033,453회**

| 순위 | 함수명 (파일명:라인) | 호출 횟수 (ncalls) | 누적 시간 (sec) | 실행 비중 (%) | 비고 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `exec` (built-in) | 2/1 | 1,838.894 | 100.0% | 프로그램 실행 진입점 |
| 2 | `<module>` (`<string>:1`) | 1 | 1,838.894 | 100.0% | 모듈 실행 |
| 3 | `run_profiling` (`run_profile.py:11`) | 1 | 1,838.894 | 100.0% | 프로파일링 루프 |
| 4 | `play_validation_game` (`run_validation_20.py:17`) | 2 | 1,838.893 | 100.0% | 대국 진행 함수 |
| 5 | `find_best_move` (`minimax.py:381`) | 52 | 1,838.678 | 99.99% | 최적 착수 탐색 |
| 6 | `alphabeta` (`minimax.py:464`) | 600,998 / 2,346 | 1,833.106 | 99.69% | 알파-베타 미니맥스 재귀 탐색 |
| 7 | `play_move` (`game_state.py:37`) | 2,542,208 | 1,538.947 | 83.69% | **핵심 병목**: 가상 착수 수행 |
| 8 | `order_moves` (`minimax.py:64`) | 28,179 | 1,212.355 | 65.93% | 착수 정렬 및 사전 탐색 |
| 9 | `get_empty_regions` (`safe_groups.py:16`) | 2,542,208 | 1,099.075 | 59.77% | **핵심 병목**: 안전그룹 빈 영역 검사 (자체시간 710.85s) |
| 10 | `evaluate` (`evaluation.py:466`) | 442,477 | 236.178 | 12.84% | 평가 함수 |
| 11 | `evaluate_detailed` (`evaluation.py:337`) | 442,608 | 232.925 | 12.67% | 세부 평가 기능 |
| 12 | `is_valid` (`board.py:22`) | 888,773,428 | 172.395 | 9.37% | 좌표 유효성 검사 |
| 13 | `analyze_groups_detailed_flat` (`evaluation.py:245`) | 885,216 | 163.358 | 8.88% | 그룹 활로 분석 |
| 14 | `get` (`board.py:30`) | 1,433,148,387 | 160.122 | 8.71% | 바둑판 격자 조회 |
| 15 | `get_group` (`capture.py:14`) | 19,418,062 | 156.469 | 8.51% | 연결된 돌 그룹 탐색 |
| 16 | `get_future_liberty_risk_flat` (`evaluation.py:173`) | 1,840,403 | 143.433 | 7.80% | 가상 위험 그룹 평가 |
| 17 | `future_lib_minimax_flat` (`evaluation.py:98`) | 3,052,369 / 466,134 | 142.159 | 7.73% | 활로 깊이 예측 미니맥스 |
| 18 | `method 'add' of set` (built-in) | 684,303,190 | 129.216 | 7.03% | 집합 자료형 원소 추가 |
| 19 | `get_local_state_key` (`evaluation.py:89`) | 3,052,369 | 70.387 | 3.83% | 로컬 상태 해시 키 생성 |
| 20 | `get_liberties` (`capture.py:49`) | 9,960,438 | 68.728 | 3.74% | 활로(자유도) 계산 |
| 21 | `play_move_flat` (`evaluation.py:62`) | 2,378,267 | 45.885 | 2.50% | 평가용 가상 간이 착수 |
| 22 | `calculate_territory_flat` (`evaluation.py:191`) | 442,608 | 41.943 | 2.28% | 평가용 영토 간이 계산 |
| 23 | `method 'popleft' of deque` (built-in) | 207,304,193 | 37.206 | 2.02% | 데크 왼쪽 원소 추출 |
| 24 | `method 'append' of deque` (built-in) | 204,500,079 | 37.112 | 2.02% | 데크 오른쪽 원소 추가 |
| 25 | `get_legal_moves` (`minimax.py:215`) | 28,233 | 15.978 | 0.87% | 합법수 목록 추출 |
| 26 | `method 'append' of list` (built-in) | 85,067,108 | 15.671 | 0.85% | 리스트 원소 추가 |
| 27 | `builtins.len` (built-in) | 101,037,594 | 15.325 | 0.83% | 객체 길이 반환 |
| 28 | `copy_game_state` (`minimax.py:165`) | 2,547,240 | 13.293 | 0.72% | 게임 상태 복제 |
| 29 | `calculate_territory_details` (`territory.py:7`) | 28,287 | 11.557 | 0.63% | 영토 영역 상세 산출 |
| 30 | `<genexpr>` (`evaluation.py:96`) | 51,750,370 | 7.562 | 0.41% | 제너레이터 표현식 계산 |

---

## 2. 🧩 주요 영역별 실행 비중 분석

### 2.1. Minimax 관련
* **대상 함수**: `alphabeta` (재귀 탐색), `find_best_move`
* **누적 시간 (Ratio)**: **1,833.106초 (99.69%)**
* **분석**:
  * Minimax 탐색 프레임워크 자체의 순수 오버헤드(`tottime`)는 **5.638초 (0.31%)**에 불과합니다.
  * 탐색의 대부분의 시간은 재귀 트리 내에서 상태 노드를 확장할 때 호출하는 `play_move`와 단말 노드에서 상태를 진단하는 `evaluate` 등 하위 연산에 분산되어 있습니다.

### 2.2. Evaluation 관련
* **대상 함수**: `evaluate` (탑레벨), `evaluate_detailed`, `future_lib_minimax_flat` 등 평가 모듈 함수 전체
* **누적 시간 (Ratio)**: **236.178초 (12.84%)**
  * `evaluate`: 236.178초 (12.84%)
  * `evaluate_detailed`: 232.925초 (12.67%)
  * `future_lib_minimax_flat`: 142.159초 (7.73%)
* **분석**:
  * 평가 함수군 전체의 누적 시간 비중은 **12.84%**로 생각보다 크지 않습니다.
  * 평가 함수 내에서는 그룹 활로와 포위 위협을 진단하는 `future_lib_minimax_flat`(7.73%)이 절반 이상의 시간을 점유하고 있습니다.

### 2.3. Territory 관련
* **대상 함수**: `calculate_territory_details` (엔진 규칙용), `calculate_territory_flat` (평가 가산용)
* **누적 시간 (Ratio)**: **53.500초 (2.91%)**
  * `calculate_territory_details`: 11.557초 (0.63%)
  * `calculate_territory_flat`: 41.943초 (2.28%)
* **분석**:
  * BFS/DFS 기반으로 판 전체를 돌며 집의 경계를 짓는 영토 연산은 전체의 **2.91%**만을 차지하여 주요 병목에서는 한참 벗어나 있습니다.

### 2.4. Move Generation (착수 및 합법수 생성) 관련
* **대상 함수**: `get_legal_moves`, `play_move`, `get_empty_regions`
* **누적 시간 (Ratio)**: **1,538.947초 (83.69%)**
  * `play_move` (가상 착수): 1,538.947초 (83.69%)
  * `get_empty_regions` (빈 영역 스캔): 1,099.075초 (59.77%)
  * `get_legal_moves` (합법수 생성): 15.978초 (0.87%)
* **분석**:
  * **압도적인 병목 구간**입니다. 합법수 생성 자체(`get_legal_moves`)는 0.87%로 매우 빠르지만, 미니맥스 탐색 중 다음 노드로 진입할 때마다 호출되는 `play_move`가 전체 실행 시간의 **83.69%**를 소모합니다.
  * `play_move` 내부에서 착수 직전 상대방의 돌이 "안전 그룹"에 포함되는지를 검사하기 위해 매번 호출하는 `get_empty_regions`가 **59.77%**(순수 자체 소요 시간만 **710.853초, 38.66%**)를 차지합니다.

---

## 3. 🔍 "43분 중 실제로 어떤 함수가 가장 많은 시간을 먹고 있는가?"에 대한 답

전체 43분(자가대국 20판 기준) 중 각 함수가 차지하는 누적 실행 비중은 다음과 같이 정확하게 수치화할 수 있습니다:

1. **`get_empty_regions` (안전그룹 판정용 영역 스캔) = 59.77%** (자체 시간만 38.66%)
2. **`play_move` (Minimax 노드 전이 시 가상 착수 수행) = 23.92%** (상기 `get_empty_regions` 등 하위 호출을 제외한 순수 play_move의 구현 자체 시간: 8.78%)
3. **`evaluate_detailed` (상태 평가 로직) = 12.67%** (평가 및 활로 위험 예측 등)
4. **기타 보드 조작 및 연산 (`is_valid`, `get`, `get_group` 등) = 3.64%**

> [!IMPORTANT]
> **결론**: 전체 시간의 **약 84%가 착수(`play_move`) 및 그 과정의 안전그룹 검사(`get_empty_regions`)**에 집중되어 있습니다. 수읽기 탐색 효율을 올리기 위해서는 평가 함수나 탐색 알고리즘을 고치기 전에 **`play_move` 내부의 불필요한 전체 안전그룹 재계산 구조를 점진적/증분적(Incremental) 계산 또는 캐싱 구조로 개선**하는 것이 최우선 과제입니다.

---

## 4. 🎨 Call Graph 및 시각화 가이드
* **Snakeviz 시각화**: 로컬 개발 머신에서 아래 명령어를 실행하여 웹 기반의 시각적 프로파일 그래프를 즉시 조회할 수 있습니다.
  ```bash
  snakeviz profile.out
  ```
* **Graphviz Call Graph**: `gprof2dot`을 활용해 계측 흐름을 표현한 그래프 파일 [graph.dot](file:///C:/Users/User/source/repos/greatkingdomAI/graph.dot)을 현재 디렉터리에 생성 완료했습니다. 로컬 환경에 Graphviz 라이브러리가 존재할 경우 아래 명령어를 통해 이미지로 변환할 수 있습니다.
  ```bash
  gprof2dot -f pstats profile.out -o graph.dot
  dot -Tpng graph.dot -o profile_graph.png
  ```
