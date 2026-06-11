import io
from contextlib import redirect_stdout
import random
import unittest
from ai.minimax import find_best_move
from engine.board import BLUE, EMPTY, ORANGE
from engine.game_state import GameState


class TestAIBiasExperiment(unittest.TestCase):

    def test_experiment_2_orange_first(self):
        print("\n=== STARTING EXPERIMENT 2: ORANGE FIRST (10 GAMES) ===")
        results = []
        orange_wins = 0

        for game_idx in range(1, 11):
            game = GameState()
            game.current_player = ORANGE  # ORANGE 선공 강제

            move_count = 0
            max_moves = 150

            while not game.game_over and move_count < max_moves:
                # stdout 리다이렉션으로 minimax 내부의 디버그 출력 로그를 억제합니다.
                f = io.StringIO()
                with redirect_stdout(f):
                    move = find_best_move(game, depth=2)

                if move == "pass":
                    game.play_pass()
                else:
                    game.play_move(move[0], move[1])
                move_count += 1

            if game.consecutive_passes >= 2:
                reason = "Territory (Komi)"
            elif game.game_over and game.winner is not None:
                reason = "Capture"
            else:
                reason = "Max moves limit"

            winner_str = (
                "BLUE (Player 1)"
                if game.winner == BLUE
                else (
                    "ORANGE (Player 2)" if game.winner == ORANGE else "Draw"
                )
            )

            if game.winner == ORANGE:
                orange_wins += 1

            print(
                f"Exp 2 | Game {game_idx:02d} | Winner: {winner_str:<18} | Moves: {move_count:<3} | Reason: {reason}"
            )
            results.append(game.winner)

        print(
            f"=== EXPERIMENT 2 COMPLETED | ORANGE Wins: {orange_wins}/10 ==="
        )

    def test_experiment_3_random_first_move(self):
        print(
            "\n=== STARTING EXPERIMENT 3: RANDOM FIRST MOVE (20 GAMES) ==="
        )
        results = []
        random.seed(42)  # 실험 재현을 위해 랜덤 시드 고정

        for game_idx in range(1, 21):
            game = GameState()
            size = game.board.size
            move_count = 0
            max_moves = 150
            first_move_coords = None

            # 첫 턴만 임의의 빈 공간 중 하나에 무작위 착수 강제
            empty_cells = []
            for r in range(size):
                for c in range(size):
                    if game.board.get(r, c) == EMPTY:
                        empty_cells.append((r, c))

            first_move_coords = random.choice(empty_cells)
            game.play_move(first_move_coords[0], first_move_coords[1])
            move_count += 1

            # 두 번째 턴부터는 정상적으로 Minimax 작동
            while not game.game_over and move_count < max_moves:
                # stdout 리다이렉션으로 minimax 내부의 디버그 출력 로그를 억제합니다.
                f = io.StringIO()
                with redirect_stdout(f):
                    move = find_best_move(game, depth=2)

                if move == "pass":
                    game.play_pass()
                else:
                    game.play_move(move[0], move[1])
                move_count += 1

            if game.consecutive_passes >= 2:
                reason = "Territory (Komi)"
            elif game.game_over and game.winner is not None:
                reason = "Capture"
            else:
                reason = "Max moves limit"

            winner_str = (
                "BLUE (Player 1)"
                if game.winner == BLUE
                else (
                    "ORANGE (Player 2)" if game.winner == ORANGE else "Draw"
                )
            )

            print(
                f"Exp 3 | Game {game_idx:02d} | First Move: {first_move_coords} | Winner: {winner_str:<18} | Moves: {move_count:<3} | Reason: {reason}"
            )
            results.append(
                (first_move_coords, game.winner, move_count, reason)
            )

        print("=== EXPERIMENT 3 COMPLETED ===\n")


if __name__ == "__main__":
    unittest.main()
