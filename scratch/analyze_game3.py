import sys
import os

sys.path.append(r"C:\Users\User\source\repos\greatkingdomAI")

from engine.game_state import GameState
from engine.board import BLUE, ORANGE, NEUTRAL, EMPTY
from ai.evaluation import evaluate_detailed
from ai.minimax import get_legal_moves, copy_game_state, alphabeta

def main():
    moves = [
        [1, 2], [2, 3], [2, 1], [1, 6], [2, 7], 
        [6, 6], [6, 2], [5, 1], [6, 0], [6, 1], 
        [7, 1], [7, 3], [7, 2], [3, 8], [0, 7], 
        "pass", 
        [7, 7], [1, 7], [1, 8], [8, 5], [8, 6], 
        [6, 8], [5, 7], [0, 3], [4, 7], [0, 5], 
        [4, 8], [3, 7], [3, 6], [1, 1], [2, 8]
    ]

    game = GameState()
    game.is_copy = True

    states = []
    states.append((0, None, None, copy_game_state(game)))

    for idx, move in enumerate(moves):
        player = game.current_player
        if move == "pass":
            game.play_pass()
        else:
            game.play_move(move[0], move[1])
        states.append((idx + 1, player, move, copy_game_state(game)))

    print("=== GAME 3 MOVES TRACE ===")
    
    # 1. 15수 직후 (16수 ORANGE 차례 직전) 상태 분석
    m_idx, player, move, state_15 = states[15]
    print(f"\n[Move 15] Played by {'BLUE (Human)' if player==BLUE else 'ORANGE (AI)'}: {move}")
    state_15.board.display()
    
    # ORANGE 관점의 평가 세부 정보
    details_15 = evaluate_detailed(state_15.board, ORANGE)
    print("ORANGE perspective details at Move 15:")
    for k, v in details_15.items():
        print(f"  {k}: {v:.1f}")

    # ORANGE가 16수에서 왜 pass를 했는지 minimax로 탐색 후보 확인
    print("\n[Move 16] ORANGE Decision Analysis:")
    legal_moves_16 = get_legal_moves(state_15)
    move_scores = []
    for mv in legal_moves_16:
        ns = copy_game_state(state_15)
        try:
            if mv == "pass":
                ns.play_pass()
            else:
                ns.play_move(mv[0], mv[1])
        except ValueError:
            continue
        # Depth 3 탐색
        score = alphabeta(ns, 2, -float("inf"), float("inf"), False, ORANGE)
        move_scores.append((mv, score))
    
    move_scores.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 moves for ORANGE at Move 16:")
    for idx_c, (mv, sc) in enumerate(move_scores[:10]):
        print(f"  #{idx_c+1:02d} | Move: {mv} | Score: {sc:.2f}")

    # 2. 30수 직후 (31수 BLUE 차례 직전) 상태 분석
    m_idx, player, move, state_30 = states[30]
    print(f"\n[Move 30] Played by {'BLUE (Human)' if player==BLUE else 'ORANGE (AI)'}: {move}")
    state_30.board.display()
    
    # 3. 31수 직후 (대국 종료) 상태 분석
    m_idx, player, move, state_31 = states[31]
    print(f"\n[Move 31] Played by {'BLUE (Human)' if player==BLUE else 'ORANGE (AI)'}: {move}")
    state_31.board.display()
    print("Winner:", "BLUE" if state_31.winner == BLUE else "ORANGE" if state_31.winner == ORANGE else "None")
    print("Game Over:", state_31.game_over)

if __name__ == "__main__":
    main()
