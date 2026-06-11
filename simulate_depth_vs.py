import io
import time
import random
import multiprocessing
from contextlib import redirect_stdout
from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from ai.minimax import find_best_move, clear_transposition_table, reset_stats


def play_vs_game(args):
    """자식 프로세스에서 실행될 Depth 2 vs Depth 3 대국"""
    game_idx, blue_depth, orange_depth = args
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
            # 현재 차례 플레이어에 매핑된 depth 사용
            curr_depth = blue_depth if game.current_player == BLUE else orange_depth
            f = io.StringIO()
            with redirect_stdout(f):
                move = find_best_move(game, depth=curr_depth)

        if move == "pass":
            game.play_pass()
        else:
            game.play_move(move[0], move[1])
        move_count += 1

    game_duration = time.time() - game_start

    # 승자 확인
    winner = game.winner
    if winner is None:
        winner = game.check_winner()

    # 종료 사유
    if game.game_over:
        if game.consecutive_passes >= 2:
            reason = "consecutive_passes"
        else:
            reason = "other"
    else:
        reason = "max_moves"

    return {
        "game_idx": game_idx,
        "blue_depth": blue_depth,
        "orange_depth": orange_depth,
        "winner": winner,
        "moves": move_count,
        "duration": game_duration,
        "reason": reason,
    }


def run_vs_simulation(num_games=50):
    print("=== GREAT KINGDOM AI DEPTH 2 vs DEPTH 3 SIMULATION ===", flush=True)

    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    print(f"Cores: {num_cores} | Using Processes: {num_processes}", flush=True)
    print(f"Total Games: {num_games}\n", flush=True)

    # 1~25판: BLUE(Depth 2) vs ORANGE(Depth 3)
    # 26~50판: BLUE(Depth 3) vs ORANGE(Depth 2)
    tasks = []
    half_games = num_games // 2
    for i in range(1, num_games + 1):
        if i <= half_games:
            tasks.append((i, 2, 3))
        else:
            tasks.append((i, 3, 2))

    start_time = time.time()

    d2_wins = 0
    d3_wins = 0
    draws = 0

    blue_wins = 0
    orange_wins = 0

    total_moves = 0
    total_duration = 0

    reasons = {"consecutive_passes": 0, "max_moves": 0, "other": 0}

    completed_games = 0

    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(play_vs_game, tasks):
            completed_games += 1

            winner = result["winner"]
            moves = result["moves"]
            duration = result["duration"]
            reason = result["reason"]
            blue_depth = result["blue_depth"]
            orange_depth = result["orange_depth"]

            total_moves += moves
            total_duration += duration
            reasons[reason] += 1

            # 선/후공 집계
            if winner == BLUE:
                blue_wins += 1
                winner_color = "BLUE"
                winner_depth = blue_depth
            elif winner == ORANGE:
                orange_wins += 1
                winner_color = "ORANGE"
                winner_depth = orange_depth
            else:
                winner_color = "DRAW"
                winner_depth = None

            # Depth 집계
            if winner_depth == 2:
                d2_wins += 1
                winner_depth_str = "Depth 2"
            elif winner_depth == 3:
                d3_wins += 1
                winner_depth_str = "Depth 3"
            else:
                draws += 1
                winner_depth_str = "DRAW"

            print(
                f"[{completed_games:02d}/{num_games:02d}] Game #{result['game_idx']:02d} finished | "
                f"Config: BLUE(D{blue_depth}) vs ORANGE(D{orange_depth}) | "
                f"Winner: {winner_color}({winner_depth_str}) | Moves: {moves:3d} | Time: {duration:.2f}s | "
                f"Stats -> D2 Wins: {d2_wins} | D3 Wins: {d3_wins} | Draw: {draws}",
                flush=True,
            )

    elapsed_time = time.time() - start_time

    # 종합 결과 분석 출력
    print("\n================ VS SIMULATION RESULTS ================", flush=True)
    print(
        f"Total Simulation Time : {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)",
        flush=True,
    )
    print(f"Total Games Played    : {num_games}", flush=True)
    print("-" * 55, flush=True)
    print(
        f"Depth 2 Wins          : {d2_wins} ({d2_wins/num_games*100:.1f}%)",
        flush=True,
    )
    print(
        f"Depth 3 Wins          : {d3_wins} ({d3_wins/num_games*100:.1f}%)",
        flush=True,
    )
    print(f"Draws                 : {draws} ({draws/num_games*100:.1f}%)", flush=True)
    print("-" * 55, flush=True)
    print(
        f"BLUE Wins (First)     : {blue_wins} ({blue_wins/num_games*100:.1f}%)",
        flush=True,
    )
    print(
        f"ORANGE Wins (Second)  : {orange_wins} ({orange_wins/num_games*100:.1f}%)",
        flush=True,
    )
    print("-" * 55, flush=True)
    print(f"Average Moves per Game: {total_moves/num_games:.2f}", flush=True)
    print(f"Average Game Duration : {total_duration/num_games:.2f} seconds", flush=True)
    print("-" * 55, flush=True)
    capture_count = reasons["other"]
    territory_count = reasons["consecutive_passes"]
    print("Termination Reasons Summary:", flush=True)
    print(
        f"  - Capture Wins (Other)        : {capture_count} ({capture_count/num_games*100:.1f}%)",
        flush=True,
    )
    print(
        f"  - Territory Wins (Passes)     : {territory_count} ({territory_count/num_games*100:.1f}%)",
        flush=True,
    )
    print(
        f"  - Max Moves Reached           : {reasons['max_moves']} ({reasons['max_moves']/num_games*100:.1f}%)",
        flush=True,
    )
    print("=======================================================", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_vs_simulation(num_games=50)
