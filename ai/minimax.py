import random
from ai.evaluation import evaluate, evaluate_detailed
from ai.zobrist import get_board_hash
from engine.board import EMPTY


# Transposition Table 데이터 저장소
TRANSPOSITION_TABLE = {}

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

# UI 통신용 마지막 AI 의사결정 기록
LAST_AI_DECISION = {"move": None, "score": 0.0, "depth": 3}

# 성능 측정용 글로벌 메트릭
STATS = {"nodes_visited": 0, "cutoffs": 0}


def reset_stats():
    """성능 측정을 위해 계측 데이터를 초기화합니다."""
    STATS["nodes_visited"] = 0
    STATS["cutoffs"] = 0


def clear_transposition_table():
    """대국 간 공정하거나 독립적인 벤치마크를 위해 치환표를 청소합니다."""
    TRANSPOSITION_TABLE.clear()


def lookup_entry(hash_key, depth, alpha, beta):
    """치환표에서 캐싱된 평가값이 있는지 유효성을 검사하여 반환합니다."""
    if hash_key in TRANSPOSITION_TABLE:
        entry = TRANSPOSITION_TABLE[hash_key]
        # 더 깊거나 같은 깊이로 탐색된 내용만 활용 가능
        if entry["depth"] >= depth:
            val = entry["value"]
            if entry["flag"] == EXACT:
                return val
            elif entry["flag"] == LOWERBOUND:
                if val >= beta:
                    return val
            elif entry["flag"] == UPPERBOUND:
                if val <= alpha:
                    return val
    return None


def store_entry(hash_key, depth, value, flag, best_move=None):
    """현재 노드의 탐색 깊이와 정확도 플래그, 최적의 수순을 치환표에 등록합니다."""
    if (
        hash_key not in TRANSPOSITION_TABLE
        or TRANSPOSITION_TABLE[hash_key]["depth"] <= depth
    ):
        TRANSPOSITION_TABLE[hash_key] = {
            "depth": depth,
            "value": value,
            "flag": flag,
            "best_move": best_move,
        }


def order_moves(state, legal_moves, depth, maximizing_player, target_player):
    """Alpha-Beta Pruning 효율을 높이기 위해 착수 후보 목록을 가볍고 빠르게 가중치 정렬합니다. (evaluate 호출 없음)"""
    if len(legal_moves) <= 1:
        return legal_moves

    hash_key = state.hash_val
    best_move = None

    # 1. 치환표(TT)에 저장된 최적의 수순이 있으면 최우선순위로 설정
    if hash_key in TRANSPOSITION_TABLE:
        best_move = TRANSPOSITION_TABLE[hash_key].get("best_move")

    # 각 move별 가중치 매기기
    from engine.capture import get_group, get_liberties
    from engine.board import EMPTY
    
    scored_moves = []
    
    # 2. 착수 전 내 돌들의 위험 그룹(자유도가 1 또는 2)의 이웃 활로 좌표들을 파악 (Defensive/Escape용)
    my_color = state.current_player
    opp_color = state.opponent()
    board = state.board
    
    danger_liberties = set()
    checked_my = set()
    for r in range(board.size):
        for c in range(board.size):
            if board.get(r, c) == my_color and (r, c) not in checked_my:
                g = get_group(board, r, c)
                checked_my.update(g)
                libs = get_liberties(board, g)
                if len(libs) <= 2:
                    danger_liberties.update(libs) # 내 위험한 돌들의 활로 좌표 모음

    for move in legal_moves:
        if move == best_move:
            continue
        if move == "pass":
            # 패스는 가장 뒤로 보냄
            scored_moves.append((move, -999999))
            continue
        
        # 기본 점수는 격자 중앙 지향성 (중앙에 가까울수록 가산점)
        dist_from_center = abs(move[0] - 4) + abs(move[1] - 4)
        score = 100 - dist_from_center  # 기본 점수: 92 ~ 100 점
        
        # 가상 착수 (경량 연산)
        next_state = copy_game_state(state)
        is_capture = False
        try:
            # play_move는 캡처 시 True 반환
            is_capture = next_state.play_move(move[0], move[1])
        except ValueError:
            # 둘 수 없는 곳(자충 등)은 점수를 아주 낮춤
            scored_moves.append((move, -50000))
            continue
            
        # A. 상대방을 따내 승리/캡처하는 수
        if is_capture:
            score += 50000
            
        # B. 내 위험한 돌들을 살리는 수 (Escape)
        elif move in danger_liberties:
            # 착수 후 내 돌들의 최소 자유도가 증가했는지 확인
            next_g = get_group(next_state.board, move[0], move[1])
            next_libs = get_liberties(next_state.board, next_g)
            if len(next_libs) > 2:
                score += 15000 # 확실히 탈출 성공
            else:
                score += 5000  # 연명 시도
                
        # C. 상대방을 단수(Atari) 치는 수
        # 방금 둔 자리 주변에 상대 돌이 있는지 보고, 그 상대 그룹의 자유도가 1로 조여졌는지 판단
        if not is_capture:
            opp_groups_checked = set()
            is_atari = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = move[0] + dr, move[1] + dc
                if 0 <= nr < board.size and 0 <= nc < board.size:
                    if next_state.board.get(nr, nc) == opp_color and (nr, nc) not in opp_groups_checked:
                        g = get_group(next_state.board, nr, nc)
                        opp_groups_checked.update(g)
                        libs = get_liberties(next_state.board, g)
                        if len(libs) == 1:
                            is_atari = True
                            break
            if is_atari:
                score += 10000

        scored_moves.append((move, score))

    # 점수 내림차순 정렬 (항상 최선수를 먼저 탐색하게 함)
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    
    ordered = [m for m, _ in scored_moves]
    if best_move and best_move in legal_moves:
        ordered.insert(0, best_move)

    return ordered


def copy_game_state(game_state):
    """현재 게임 상태를 안전하게 복사하여 독립적인 다음 수를 시뮬레이션할 수 있게 합니다."""
    from engine.game_state import GameState

    new_state = GameState.__new__(GameState)
    new_state.board = game_state.board.copy()
    new_state.current_player = game_state.current_player
    new_state.consecutive_passes = game_state.consecutive_passes
    new_state.winner = game_state.winner
    new_state.game_over = game_state.game_over
    new_state.is_copy = True
    new_state.hash_val = game_state.hash_val
    if hasattr(game_state, "_opponent_territory"):
        new_state._opponent_territory = game_state._opponent_territory
    return new_state


ALL_CELLS_SORTED = sorted(
    [(r, c) for r in range(9) for c in range(9)],
    key=lambda coord: abs(coord[0] - 4) + abs(coord[1] - 4)
)

def get_min_lib_info(board, player):
    """지정된 플레이어의 모든 돌 그룹을 순회하여 최소 자유도, 최소 자유도 그룹의 활로 좌표 세트, 단수(Atari) 그룹 개수를 반환합니다."""
    from engine.capture import get_group, get_liberties
    
    visited = set()
    min_lib = 99
    min_lib_coords = set()
    atari_groups_count = 0
    
    for r in range(board.size):
        for c in range(board.size):
            if board.get(r, c) == player and (r, c) not in visited:
                g = get_group(board, r, c)
                visited.update(g)
                libs = get_liberties(board, g)
                num_libs = len(libs)
                
                if num_libs < min_lib:
                    min_lib = num_libs
                    min_lib_coords = set(libs)
                elif num_libs == min_lib:
                    min_lib_coords.update(libs)
                    
                if num_libs == 1:
                    atari_groups_count += 1
                    
    return min_lib, min_lib_coords, atari_groups_count

def get_legal_moves(game_state):
    """현재 보드 상태에서 착수 가능한 모든 합법 수 좌표 목록 및 'pass' 행동을 반환합니다.
    (상대 플레이어의 영토 내부 착수는 제한됩니다.)
    """
    from engine.territory import calculate_territory_details
    from engine.board import BLUE, ORANGE, EMPTY

    grid = game_state.board.grid
    opponent = game_state.opponent()
    
    # 상대 영토 계산 (캐싱 적용)
    if hasattr(game_state, "_opponent_territory") and game_state._opponent_territory is not None:
        opponent_territory = game_state._opponent_territory
    else:
        _, _, blue_coords, orange_coords = calculate_territory_details(game_state.board)
        opponent_territory = set(orange_coords) if opponent == ORANGE else set(blue_coords)
        game_state._opponent_territory = opponent_territory

    moves = []

    for r, c in ALL_CELLS_SORTED:
        if grid[r][c] == EMPTY:
            if (r, c) not in opponent_territory:
                moves.append((r, c))

    # 마지막으로 pass를 탐색 후보에 추가
    moves.append("pass")

    # 강제 후보 필터링 적용 (위험 탈출 수 강제 - 양방 플레이어 모두에 적용하여 전술 일관성 확보)
    if not getattr(game_state, "is_copy", False):
        current_player = game_state.current_player
        
        current_min_lib, my_min_lib_coords, _ = get_min_lib_info(game_state.board, current_player)

        # ----------------- [1단계: 최상위 공격 필터링] -----------------
        # 즉시 상대 돌을 캡처하거나 양단수를 치는 찬스가 있다면 즉각 공격을 감행
        capture_moves = []
        double_atari_moves = []
        atari_moves = []
        other_moves = []
        
        for move in moves:
            if move == "pass":
                other_moves.append(move)
                continue
            next_state = copy_game_state(game_state)
            try:
                # play_move는 캡처 시 True 반환
                is_capture = next_state.play_move(move[0], move[1])
            except ValueError:
                continue
            
            # A. 즉시 캡처 (게임 승리)
            if next_state.game_over and next_state.winner == current_player:
                capture_moves.append(move)
                continue
                
            opp_min_lib, _, opp_atari_groups = get_min_lib_info(next_state.board, opponent)
            
            if is_capture:
                capture_moves.append(move)
            elif opp_atari_groups >= 2:
                double_atari_moves.append(move)
            elif opp_min_lib == 1:
                atari_moves.append(move)
            else:
                other_moves.append(move)
                
        if capture_moves:
            return capture_moves
        if double_atari_moves:
            return double_atari_moves

        # ----------------- [2단계: 생존 수비 비상 모드] -----------------
        # 즉시 승리하는 공격 찬스가 없을 때, 내 돌의 사활/포위 위기를 최우선 방어
        if current_min_lib <= 2:
            filtered_moves = []
            really_safe_moves = []
            
            # 루프를 돌 대상 수순을 대마의 인접 활로 좌표로 한정 (없으면 폴백으로 전체 moves)
            target_moves = [m for m in moves if m in my_min_lib_coords] if my_min_lib_coords else moves
            
            if current_min_lib == 1:
                # 1단계: 자유도가 2 이상으로 회복되는 수순 탐색
                for move in target_moves:
                    if move == "pass":
                        continue
                    next_state = copy_game_state(game_state)
                    try:
                        next_state.play_move(move[0], move[1])
                    except ValueError:
                        continue
                    next_min_lib, _, _ = get_min_lib_info(next_state.board, current_player)
                    if next_min_lib > 1:
                        filtered_moves.append(move)
                        
                        # FL3 판단을 위한 최소 호출
                        grid_flat = next_state.board.grid[0] + next_state.board.grid[1] + next_state.board.grid[2] + next_state.board.grid[3] + next_state.board.grid[4] + next_state.board.grid[5] + next_state.board.grid[6] + next_state.board.grid[7] + next_state.board.grid[8]
                        from engine.capture import get_group
                        g = get_group(next_state.board, move[0], move[1])
                        group_set = frozenset([r*9 + c for r, c in g])
                        from ai.evaluation import get_future_liberty_risk_flat
                        _, _, fl3 = get_future_liberty_risk_flat(grid_flat, group_set, current_player, next_min_lib)
                        if fl3 > 0:
                            really_safe_moves.append(move)
                
                # 2단계: 1단계가 없다면, 자유도 1로 연명이라도 하는 수순 탐색 (선 뻗기 등 수용)
                if not filtered_moves:
                    for move in target_moves:
                        if move == "pass":
                            continue
                        next_state = copy_game_state(game_state)
                        try:
                            next_state.play_move(move[0], move[1])
                        except ValueError:
                            continue
                        next_min_lib, _, _ = get_min_lib_info(next_state.board, current_player)
                        if next_min_lib == 1:
                            filtered_moves.append(move)
                            
                            grid_flat = next_state.board.grid[0] + next_state.board.grid[1] + next_state.board.grid[2] + next_state.board.grid[3] + next_state.board.grid[4] + next_state.board.grid[5] + next_state.board.grid[6] + next_state.board.grid[7] + next_state.board.grid[8]
                            from engine.capture import get_group
                            g = get_group(next_state.board, move[0], move[1])
                            group_set = frozenset([r*9 + c for r, c in g])
                            from ai.evaluation import get_future_liberty_risk_flat
                            _, _, fl3 = get_future_liberty_risk_flat(grid_flat, group_set, current_player, next_min_lib)
                            if fl3 > 0:
                                really_safe_moves.append(move)
                            
            elif current_min_lib == 2:
                # 자유도 2 상태에서는, 착수 후 자유도가 2 이상으로 유지되거나 늘어나는 수순만 후보로 삼아 갇힘 회피
                for move in target_moves:
                    if move == "pass":
                        continue
                    next_state = copy_game_state(game_state)
                    try:
                        next_state.play_move(move[0], move[1])
                    except ValueError:
                        continue
                    next_min_lib, _, _ = get_min_lib_info(next_state.board, current_player)
                    if next_min_lib >= 2:
                        filtered_moves.append(move)
                        
                        grid_flat = next_state.board.grid[0] + next_state.board.grid[1] + next_state.board.grid[2] + next_state.board.grid[3] + next_state.board.grid[4] + next_state.board.grid[5] + next_state.board.grid[6] + next_state.board.grid[7] + next_state.board.grid[8]
                        from engine.capture import get_group
                        g = get_group(next_state.board, move[0], move[1])
                        group_set = frozenset([r*9 + c for r, c in g])
                        from ai.evaluation import get_future_liberty_risk_flat
                        _, _, fl3 = get_future_liberty_risk_flat(grid_flat, group_set, current_player, next_min_lib)
                        if fl3 > 0:
                            really_safe_moves.append(move)
            
            if filtered_moves:
                if really_safe_moves:
                    return filtered_moves
                else:
                    other_all = [m for m in moves if m not in filtered_moves]
                    return filtered_moves + other_all

        # ----------------- [3단계: 일반 상황 - 단수 우선 정렬] -----------------
        if atari_moves:
            return atari_moves + other_moves

    return moves


def find_best_move(game_state, depth=2):
    """Alpha-Beta Minimax 알고리즘을 사용해 현재 플레이어의 최적의 수((r, c) 또는 'pass')를 찾습니다.

    탐색된 후보들 중 상위 10개 수의 항목별 평가 점수를 출력하여 의사결정 로그를 생성합니다.
    """
    from ai.evaluation import clear_future_liberty_cache
    clear_future_liberty_cache()
    from engine.safe_groups import clear_empty_regions_cache
    clear_empty_regions_cache()
    
    target_player = game_state.current_player

    if game_state.game_over:
        return "pass"

    legal_moves = get_legal_moves(game_state)
    move_scores = []

    # root 노드이므로 직접 자식 노드 평가를 먼저 수행
    alpha = -float("inf")
    beta = float("inf")

    for move in legal_moves:
        next_state = copy_game_state(game_state)
        try:
            if move == "pass":
                next_state.play_pass()
            else:
                next_state.play_move(move[0], move[1])
        except ValueError:
            # 유효하지 않은 착수의 경우(예외 발생 시) 스킵
            continue

        score = alphabeta(
            next_state, depth - 1, alpha, beta, False, target_player
        )
        move_scores.append((move, score, next_state))

        alpha = max(alpha, score)

    # 점수 높은 순으로 정렬
    move_scores.sort(key=lambda x: x[1], reverse=True)

    # 상위 10개 후보 추출
    top_candidates = move_scores[:10]

    # 디버그 상세 로그 출력 (가상 시뮬레이션 복사본이 아닐 때만 출력)
    # 콘솔 출력 금지 정책 적용 (DEBUG = False로 비활성화)
    if False:
        player_name = (
            "BLUE (Player 1)"
            if target_player == 1
            else "ORANGE (Player 2)"
        )
        print(f"\n--- AI Think Log | Actor: {player_name} ---")
        for idx, (move, total_score, next_state) in enumerate(top_candidates):
            details = evaluate_detailed(next_state.board, target_player)

            move_str = f"Move {move}" if move != "pass" else "Move pass"
            print(f"  Candidate #{idx+1:02d} | {move_str}")
            print(f"    Territory   : {details['Territory']:.1f}")
            print(f"    Liberty     : {details['Liberty']:.1f}")
            print(f"    Connectivity: {details['Connectivity']:.1f}")
            print(f"    Center      : {details['Center']:.1f}")
            print(f"    Total Score : {total_score:.1f}")
            print()
        print("------------------------------------------\n")

    if top_candidates:
        best_score = top_candidates[0][1]
        # 동점인 최적의 수들을 모두 모음 (부동 소수점 오차 감안)
        best_candidates = [move for move, score, _ in move_scores if abs(score - best_score) < 1e-7]
        chosen_move = random.choice(best_candidates)
        
        # UI 연동용 글로벌 변수에 결정 정보 기록 (가상 카피본 시뮬레이션 중이 아닐 때만 기록)
        if not getattr(game_state, "is_copy", False):
            LAST_AI_DECISION["move"] = chosen_move
            LAST_AI_DECISION["score"] = best_score
            LAST_AI_DECISION["depth"] = depth
            
        return chosen_move
    return "pass"



def alphabeta(state, depth, alpha, beta, maximizing_player, target_player):
    """Zobrist 해시 및 치환표(Transposition Table)를 연동한 Alpha-Beta Pruning 탐색 함수입니다."""
    # 노드 방문수 기록 증가
    STATS["nodes_visited"] += 1

    # 1. Zobrist 해시 키 생성
    hash_key = state.hash_val

    # 2. 치환표 캐시 검사
    cached_val = lookup_entry(hash_key, depth, alpha, beta)
    if cached_val is not None:
        return cached_val

    # 기저 조건: 최대 깊이에 도달했거나 게임이 끝난 경우
    if depth == 0 or state.game_over:
        val = 0
        if state.game_over:
            if state.winner == target_player:
                val = 1000000 + depth  # 더 빠른 승리 선호
            elif state.winner is not None:
                val = -1000000 - depth  # 더 느린 패배 선호
            else:
                return 0
        else:
            val = evaluate(state.board, target_player)

        # 기저 평가값 치환표 등록
        store_entry(hash_key, depth, val, EXACT)
        return val

    legal_moves = get_legal_moves(state)
    legal_moves = order_moves(state, legal_moves, depth, maximizing_player, target_player)
    original_alpha = alpha

    best_move = None

    if maximizing_player:
        max_eval = -float("inf")
        for move in legal_moves:
            next_state = copy_game_state(state)
            try:
                if move == "pass":
                    next_state.play_pass()
                else:
                    next_state.play_move(move[0], move[1])
            except ValueError:
                continue

            eval_val = alphabeta(
                next_state, depth - 1, alpha, beta, False, target_player
            )
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                STATS["cutoffs"] += 1
                break  # Beta 가지치기

        # 탐색 완료 후 치환표 등록 플래그 판정
        if max_eval <= original_alpha:
            flag = UPPERBOUND
        elif max_eval >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        store_entry(hash_key, depth, max_eval, flag, best_move)

        return max_eval

    else:
        min_eval = float("inf")
        for move in legal_moves:
            next_state = copy_game_state(state)
            try:
                if move == "pass":
                    next_state.play_pass()
                else:
                    next_state.play_move(move[0], move[1])
            except ValueError:
                continue

            eval_val = alphabeta(
                next_state, depth - 1, alpha, beta, True, target_player
            )
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                STATS["cutoffs"] += 1
                break  # Alpha 가지치기

        # 탐색 완료 후 치환표 등록 플래그 판정
        if min_eval <= original_alpha:
            flag = UPPERBOUND
        elif min_eval >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        store_entry(hash_key, depth, min_eval, flag, best_move)

        return min_eval
