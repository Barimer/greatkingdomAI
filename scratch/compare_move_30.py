import sys
import os

sys.path.append(r"C:\Users\User\source\repos\greatkingdomAI")

from engine.game_state import GameState
from engine.board import BLUE, ORANGE, NEUTRAL, EMPTY
from ai.minimax import get_legal_moves, copy_game_state, alphabeta

def main():
    moves = [
        [1, 2], [2, 3], [2, 1], [1, 6], [2, 7], 
        [6, 6], [6, 2], [5, 1], [6, 0], [6, 1], 
        [7, 1], [7, 3], [7, 2], [3, 8], [0, 7], 
        "pass", 
        [7, 7], [1, 7], [1, 8], [8, 5], [8, 6], 
        [6, 8], [5, 7], [0, 3], [4, 7], [0, 5], 
        [4, 8], [3, 7], [3, 6]  # 1수부터 29수까지
    ]

    game = GameState()
    game.is_copy = False

    for move in moves:
        if move == "pass":
            game.play_pass()
        else:
            game.play_move(move[0], move[1])

    # 29수 직후 상태 (30수 ORANGE 차례 직전)
    print("--- MINIMAX DEPTH 3 AT MOVE 30 ---")
    
    legal_moves = get_legal_moves(game)
    move_scores = []
    
    for move in legal_moves:
        next_state = copy_game_state(game)
        try:
            if move == "pass":
                next_state.play_pass()
            else:
                next_state.play_move(move[0], move[1])
        except ValueError:
            continue
            
        score = alphabeta(next_state, 2, -float("inf"), float("inf"), False, ORANGE)
        move_scores.append((move, score))
        
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 candidates for ORANGE at Move 30:")
    for idx, (mv, sc) in enumerate(move_scores[:10]):
        print(f"  #{idx+1:02d} | Move: {mv} | Score: {sc:.2f}")

    # (2, 8)과 (1, 1)의 점수와 랭크 확인
    for i, (mv, sc) in enumerate(move_scores):
        if mv == (2, 8):
            print(f"  [2, 8] Rank: {i+1} | Score: {sc:.2f}")
        if mv == (1, 1):
            print(f"  [1, 1] Rank: {i+1} | Score: {sc:.2f}")

if __name__ == "__main__":
    main()
