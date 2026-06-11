import io
import time
from contextlib import redirect_stdout
from ai.minimax import (
    STATS,
    clear_transposition_table,
    find_best_move,
    reset_stats,
)
from engine.game_state import GameState


def run_benchmark_for_depth(depth, num_games):
    total_game_durations = []
    total_turn_durations = []
    total_nodes_visited = []
    total_cutoffs = []

    print(
        f"\n>>> Running benchmark for Depth {depth} (over {num_games} games)...",
        flush=True,
    )

    for game_idx in range(1, num_games + 1):
        clear_transposition_table()
        reset_stats()

        game = GameState()
        game.is_copy = True

        game_start_time = time.time()
        move_count = 0
        max_moves = 100

        print(f"  Starting Game {game_idx}/{num_games}...", flush=True)

        while not game.game_over and move_count < max_moves:
            prev_nodes = STATS["nodes_visited"]
            prev_cutoffs = STATS["cutoffs"]

            turn_start = time.time()

            # minimax 내부 디버그 출력은 억제
            f = io.StringIO()
            with redirect_stdout(f):
                move = find_best_move(game, depth=depth)

            turn_duration = time.time() - turn_start
            total_turn_durations.append(turn_duration)

            nodes_in_turn = STATS["nodes_visited"] - prev_nodes
            cutoffs_in_turn = STATS["cutoffs"] - prev_cutoffs

            # 각 수순의 상태를 버퍼링 없이 즉시 실시간 강제 출력!
            player_str = (
                "BLUE" if game.current_player == 1 else "ORANGE"
            )
            print(
                f"    Move {move_count+1:03d} | Player: {player_str:<6} | "
                f"Search: {turn_duration:.4f}s | Nodes: {nodes_in_turn:<5} | Cutoffs: {cutoffs_in_turn:<5} | Decision: {move}",
                flush=True,
            )

            if move == "pass":
                game.play_pass()
            else:
                game.play_move(move[0], move[1])
            move_count += 1

        game_duration = time.time() - game_start_time
        total_game_durations.append(game_duration)

        total_nodes_visited.append(STATS["nodes_visited"])
        total_cutoffs.append(STATS["cutoffs"])

        print(
            f"  Game {game_idx}/{num_games} finished in {game_duration:.2f}s | Moves: {move_count} | Total Nodes: {STATS['nodes_visited']} | Total Cutoffs: {STATS['cutoffs']}\n",
            flush=True,
        )

    avg_turn_time = (
        sum(total_turn_durations) / len(total_turn_durations)
        if total_turn_durations
        else 0
    )
    avg_game_time = (
        sum(total_game_durations) / len(total_game_durations)
        if total_game_durations
        else 0
    )
    avg_nodes = (
        sum(total_nodes_visited) / len(total_nodes_visited)
        if total_nodes_visited
        else 0
    )
    avg_cutoffs = (
        sum(total_cutoffs) / len(total_cutoffs) if total_cutoffs else 0
    )

    return {
        "depth": depth,
        "avg_turn_time": avg_turn_time,
        "avg_game_time": avg_game_time,
        "avg_nodes": avg_nodes,
        "avg_cutoffs": avg_cutoffs,
    }


def main():
    print("=== GREAT KINGDOM AI PERFORMANCE BENCHMARK ===", flush=True)

    # 1. Depth 3 측정 (대국 수: 2판)
    result_d3 = run_benchmark_for_depth(depth=3, num_games=2)

    # 2. Depth 4 측정 (대국 수: 1판)
    result_d4 = run_benchmark_for_depth(depth=4, num_games=1)

    # 3. 종합 요약표 인쇄
    print(
        "\n================ BENCHMARK SUMMARY ================",
        flush=True,
    )
    print(
        f"{'Metric':<35} | {'Depth 3':<12} | {'Depth 4':<12}", flush=True
    )
    print("-" * 68, flush=True)
    print(
        f"{'Avg Search Time per Turn (seconds)':<35} | {result_d3['avg_turn_time']:<12.4f} | {result_d4['avg_turn_time']:<12.4f}",
        flush=True,
    )
    print(
        f"{'Avg Game Duration (seconds)':<35} | {result_d3['avg_game_time']:<12.2f} | {result_d4['avg_game_time']:<12.2f}",
        flush=True,
    )
    print(
        f"{'Avg Nodes Visited per Game':<35} | {result_d3['avg_nodes']:<12.1f} | {result_d4['avg_nodes']:<12.1f}",
        flush=True,
    )
    print(
        f"{'Avg Alpha-Beta Cutoffs per Game':<35} | {result_d3['avg_cutoffs']:<12.1f} | {result_d4['avg_cutoffs']:<12.1f}",
        flush=True,
    )
    print("===================================================", flush=True)


if __name__ == "__main__":
    main()
