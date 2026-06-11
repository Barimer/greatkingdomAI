import random
from ai.evaluation import evaluate, evaluate_detailed
from ai.zobrist import get_board_hash
from engine.board import EMPTY


# Transposition Table 데이터 저장소
TRANSPOSITION_TABLE = {}

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

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


def store_entry(hash_key, depth, value, flag):
    """현재 노드의 탐색 깊이와 정확도 플래그를 치환표에 등록합니다."""
    if (
        hash_key not in TRANSPOSITION_TABLE
        or TRANSPOSITION_TABLE[hash_key]["depth"] <= depth
    ):
        TRANSPOSITION_TABLE[hash_key] = {
            "depth": depth,
            "value": value,
            "flag": flag,
        }


def copy_game_state(game_state):
    """현재 게임 상태를 안전하게 복사하여 독립적인 다음 수를 시뮬레이션할 수 있게 합니다."""
    from engine.game_state import GameState

    new_state = GameState()
    new_state.board = game_state.board.copy()
    new_state.current_player = game_state.current_player
    new_state.consecutive_passes = game_state.consecutive_passes
    new_state.winner = game_state.winner
    new_state.game_over = game_state.game_over
    new_state.is_copy = True
    return new_state


def get_legal_moves(game_state):
    """현재 보드 상태에서 착수 가능한 모든 합법 수 좌표 목록 및 'pass' 행동을 반환합니다."""
    size = game_state.board.size
    center = size // 2
    moves = []

    for r in range(size):
        for c in range(size):
            if game_state.board.get(r, c) == EMPTY:
                moves.append((r, c))

    # 중앙 좌표와의 맨해튼 거리가 가까운 순으로 정렬
    moves.sort(
        key=lambda coord: abs(coord[0] - center) + abs(coord[1] - center)
    )

    # 마지막으로 pass를 탐색 후보에 추가
    moves.append("pass")
    return moves


def find_best_move(game_state, depth=2):
    """Alpha-Beta Minimax 알고리즘을 사용해 현재 플레이어의 최적의 수((r, c) 또는 'pass')를 찾습니다.

    탐색된 후보들 중 상위 10개 수의 항목별 평가 점수를 출력하여 의사결정 로그를 생성합니다.
    """
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
    if not getattr(game_state, "is_copy", False):
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
        return random.choice(best_candidates)
    return "pass"



def alphabeta(state, depth, alpha, beta, maximizing_player, target_player):
    """Zobrist 해시 및 치환표(Transposition Table)를 연동한 Alpha-Beta Pruning 탐색 함수입니다."""
    # 노드 방문수 기록 증가
    STATS["nodes_visited"] += 1

    # 1. Zobrist 해시 키 생성
    hash_key = get_board_hash(state.board, state.current_player)

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
    original_alpha = alpha

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
            max_eval = max(max_eval, eval_val)
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
        store_entry(hash_key, depth, max_eval, flag)

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
            min_eval = min(min_eval, eval_val)
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
        store_entry(hash_key, depth, min_eval, flag)

        return min_eval
