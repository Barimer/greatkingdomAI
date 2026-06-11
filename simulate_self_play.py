import io
import time
import random
import multiprocessing
from contextlib import redirect_stdout
from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from ai.minimax import find_best_move, clear_transposition_table, reset_stats


def play_single_game(args):
    """자식 프로세스에서 독립적으로 실행되는 단일 대국 함수"""
    game_idx, depth = args
    # 프로세스 내부 독립성 확보를 위해 매 대국마다 리셋
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
            # minimax 디버그 출력 가리기
            f = io.StringIO()
            with redirect_stdout(f):
                move = find_best_move(game, depth=depth)

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
        "winner": winner,
        "moves": move_count,
        "duration": game_duration,
        "reason": reason,
    }


def run_parallel_simulation(num_games=100, depth=2):
    print("=== GREAT KINGDOM AI PARALLEL SELF-PLAY SIMULATION ===", flush=True)
    # 가용 CPU 코어 감지 및 프로세스 수 설정 (1개는 여유를 둠)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    print(f"Detecting Cores: {num_cores} | Using Processes: {num_processes}", flush=True)
    print(f"Total Games: {num_games} | Depth: {depth}\n", flush=True)

    start_time = time.time()

    blue_wins = 0
    orange_wins = 0
    draws = 0

    total_moves = 0
    total_duration = 0

    reasons = {"consecutive_passes": 0, "max_moves": 0, "other": 0}

    # 병렬 인자 패킹
    tasks = [(i, depth) for i in range(1, num_games + 1)]

    completed_games = 0

    # 멀티프로세싱 Pool 기동
    with multiprocessing.Pool(processes=num_processes) as pool:
        # imap_unordered는 대국이 완료되는 순서대로 결과를 비동기 반환하므로 실시간 진행 감지에 최적화됨
        for result in pool.imap_unordered(play_single_game, tasks):
            completed_games += 1

            winner = result["winner"]
            moves = result["moves"]
            duration = result["duration"]
            reason = result["reason"]

            # 누적
            total_moves += moves
            total_duration += duration
            reasons[reason] += 1

            if winner == BLUE:
                blue_wins += 1
                winner_str = "BLUE"
            elif winner == ORANGE:
                orange_wins += 1
                winner_str = "ORANGE"
            else:
                draws += 1
                winner_str = "DRAW"

            # 매 판 완료 시마다 실시간 진행률 및 기본 결과 바로 출력!
            print(
                f"[{completed_games:03d}/{num_games:03d}] Game #{result['game_idx']:03d} finished | "
                f"Winner: {winner_str:<6} | Moves: {moves:3d} | Time: {duration:.2f}s | "
                f"Current Stats -> BLUE: {blue_wins} | ORANGE: {orange_wins} | DRAW: {draws}",
                flush=True,
            )

    elapsed_time = time.time() - start_time

    # 최종 결과 보고서 인쇄
    print("\n================ SIMULATION RESULTS ================", flush=True)
    print(
        f"Total Simulation Time : {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)",
        flush=True,
    )
    print(f"Total Games Played    : {num_games}", flush=True)
    print("-" * 52, flush=True)
    print(
        f"BLUE Wins (First)     : {blue_wins} ({blue_wins/num_games*100:.1f}%)",
        flush=True,
    )
    print(
        f"ORANGE Wins (Second)  : {orange_wins} ({orange_wins/num_games*100:.1f}%)",
        flush=True,
    )
    print(f"Draws                 : {draws} ({draws/num_games*100:.1f}%)", flush=True)
    print("-" * 52, flush=True)
    print(f"Average Moves per Game: {total_moves/num_games:.2f}", flush=True)
    print(
        f"Average Game Duration : {total_duration/num_games:.2f} seconds (Single Core Equiv)",
        flush=True,
    )
    print("-" * 52, flush=True)
    print("Termination Reasons Summary:", flush=True)
    print(f"  - Consecutive Passes: {reasons['consecutive_passes']}", flush=True)
    print(f"  - Max Moves Reached : {reasons['max_moves']}", flush=True)
    print(f"  - Other             : {reasons['other']}", flush=True)
    print("====================================================", flush=True)


if __name__ == "__main__":
    # Windows 환경 지원 필수
    multiprocessing.freeze_support()
    run_parallel_simulation(num_games=100, depth=2)
