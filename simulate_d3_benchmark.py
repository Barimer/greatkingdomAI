import io
import time
import random
import multiprocessing
from contextlib import redirect_stdout
from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from ai.minimax import find_best_move, clear_transposition_table, reset_stats


def play_d3_game(game_idx):
    """자식 프로세스에서 실행되는 Depth 3 vs Depth 3 단일 대국"""
    clear_transposition_table()
    reset_stats()

    game = GameState()
    game.is_copy = True

    game_start = time.time()
    move_count = 0
    max_moves = 150

    while not game.game_over and move_count < max_moves:
        if move_count == 0:
            from ai.minimax import get_legal_moves
            legal_moves = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal_moves)
        else:
            # 양측 모두 Depth 3 적용
            f = io.StringIO()
            with redirect_stdout(f):
                move = find_best_move(game, depth=3)

        if move == "pass":
            game.play_pass()
        else:
            game.play_move(move[0], move[1])
        move_count += 1

    game_duration = time.time() - game_start

    # 승자 판정
    winner = game.winner
    if winner is None:
        winner = game.check_winner()

    # 종료 사유 판정
    if game.game_over:
        if game.consecutive_passes >= 2:
            reason = "consecutive_passes"
        else:
            reason = "other"
    else:
        reason = "max_moves"

    return {
        "game_idx": game_idx,
        "winner": winner,
        "moves": move_count,
        "duration": game_duration,
        "reason": reason,
    }


def run_d3_benchmark(num_games=100):
    print("=== GREAT KINGDOM AI DEPTH 3 BENCHMARK ===", flush=True)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    print(f"Detected Cores: {num_cores} | Active Processes: {num_processes}", flush=True)
    print(f"Running {num_games} games under Depth 3 (Both Players)\n", flush=True)

    start_time = time.time()

    blue_wins = 0
    orange_wins = 0
    draws = 0

    total_moves = 0
    total_duration = 0

    reasons = {"consecutive_passes": 0, "max_moves": 0, "other": 0}

    # PASS 조기 종료 카운트 (수순 15수 이하이면서 consecutive_passes로 끝난 경우)
    early_pass_surrenders = 0

    move_lengths = []

    completed_games = 0

    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(play_d3_game, range(1, num_games + 1)):
            completed_games += 1

            winner = result["winner"]
            moves = result["moves"]
            duration = result["duration"]
            reason = result["reason"]

            total_moves += moves
            total_duration += duration
            reasons[reason] += 1
            move_lengths.append(moves)

            # PASS 조기 종료 체크 (15수 이하 영토 종국)
            if reason == "consecutive_passes" and moves <= 15:
                early_pass_surrenders += 1

            if winner == BLUE:
                blue_wins += 1
                winner_str = "BLUE"
            elif winner == ORANGE:
                orange_wins += 1
                winner_str = "ORANGE"
            else:
                draws += 1
                winner_str = "DRAW"

            print(
                f"[{completed_games:03d}/{num_games:03d}] Game #{result['game_idx']:03d} finished | "
                f"Winner: {winner_str:<6} | Moves: {moves:3d} | Time: {duration:.2f}s | "
                f"Stats -> BLUE: {blue_wins} | ORANGE: {orange_wins} | DRAW: {draws}",
                flush=True,
            )

    elapsed_time = time.time() - start_time

    avg_moves = total_moves / num_games
    min_moves = min(move_lengths) if move_lengths else 0
    max_moves_len = max(move_lengths) if move_lengths else 0

    capture_count = reasons["other"]
    territory_count = reasons["consecutive_passes"]
    max_limit_count = reasons["max_moves"]

    print("\n================ BENCHMARK FINAL RESULTS ================", flush=True)
    print(f"Total Elapsed Time    : {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)", flush=True)
    print(f"Total Games Played    : {num_games}", flush=True)
    print("-" * 57, flush=True)
    print(f"1. BLUE Wins (First)  : {blue_wins} ({blue_wins/num_games*100:.1f}%)", flush=True)
    print(f"2. ORANGE Wins (Second): {orange_wins} ({orange_wins/num_games*100:.1f}%)", flush=True)
    print(f"   Draws              : {draws} ({draws/num_games*100:.1f}%)", flush=True)
    print("-" * 57, flush=True)
    print(f"3. Capture 종료 횟수    : {capture_count} ({capture_count/num_games*100:.1f}%)", flush=True)
    print(f"4. Territory 종료 횟수  : {territory_count} ({territory_count/num_games*100:.1f}%)", flush=True)
    print(f"5. 평균 수순            : {avg_moves:.2f} 수", flush=True)
    print(f"6. 최소 수순            : {min_moves} 수", flush=True)
    print(f"7. 최대 수순            : {max_moves_len} 수", flush=True)
    print(f"8. PASS 조기 종료 횟수  : {early_pass_surrenders} 회", flush=True)
    print(f"   Max Limit 종료 횟수 : {max_limit_count} 회", flush=True)
    print("=========================================================", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_d3_benchmark(num_games=100)
