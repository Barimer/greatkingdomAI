import unittest
from ai.minimax import find_best_move
from engine.game_state import GameState


class TestAIBattle(unittest.TestCase):

    def test_ai_vs_ai_10_games(self):
        print("\n=== STARTING AI VS AI 10 GAMES ===")
        results = []
        total_games = 10

        for game_idx in range(1, total_games + 1):
            # 대국 시작 시 진행률 표시 (0% ~ 90%)
            base_progress = ((game_idx - 1) / total_games) * 100.0
            print(
                f"\n>>> [Progress: {base_progress:.1f}%] Starting Game {game_idx}/{total_games}..."
            )

            game = GameState()
            move_count = 0
            max_moves = 150

            while not game.game_over and move_count < max_moves:
                move = find_best_move(game, depth=2)

                # 각 플레이어의 수순 진행 상황 실시간 인쇄
                player_name = (
                    "BLUE (Player 1)"
                    if game.current_player == 1
                    else "ORANGE (Player 2)"
                )
                move_str = (
                    f"placed stone at {move}"
                    if move != "pass"
                    else "passed turn"
                )
                print(
                    f"  Game {game_idx:02d} | Move {move_count+1:03d} | {player_name} {move_str}"
                )

                if move == "pass":
                    game.play_pass()
                else:
                    game.play_move(move[0], move[1])
                move_count += 1

            # 종료 원인 판단
            if game.consecutive_passes >= 2:
                reason = "Territory (Komi)"
            elif game.game_over and game.winner is not None:
                reason = "Capture"
            else:
                reason = "Max moves limit"

            winner_str = (
                "BLUE (Player 1)"
                if game.winner == 1
                else (
                    "ORANGE (Player 2)" if game.winner == 2 else "Draw"
                )
            )

            # 대국 종료 시 진행률 표시 (10% ~ 100%)
            game_completed_progress = (game_idx / total_games) * 100.0
            print(
                f"\n<<< [Progress: {game_completed_progress:.1f}%] Game {game_idx:02d} Finished | Winner: {winner_str:<18} | Moves: {move_count:<3} | Reason: {reason}"
            )

            results.append(
                {
                    "game": game_idx,
                    "winner": winner_str,
                    "moves": move_count,
                    "reason": reason,
                }
            )

        print("\n=== AI VS AI BATTLE COMPLETED ===")
        self.assertEqual(len(results), 10)


if __name__ == "__main__":
    unittest.main()
