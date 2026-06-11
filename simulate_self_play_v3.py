import os
import json
import time
import random
import multiprocessing
import io
from contextlib import redirect_stdout
from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from ai.minimax import find_best_move, clear_transposition_table, reset_stats, get_legal_moves

def play_single_game_self_play(args):
    game_idx, depth = args
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    game.is_copy = True
    
    moves_history = []
    move_count = 0
    max_moves = 150
    
    t0 = time.time()
    while not game.game_over and move_count < max_moves:
        if move_count == 0:
            legal_moves = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal_moves)
        else:
            f = io.StringIO()
            with redirect_stdout(f):
                move = find_best_move(game, depth=depth)
                
        moves_history.append(move)
        
        if move == "pass":
            game.play_pass()
        else:
            game.play_move(move[0], move[1])
            
        move_count += 1
        # 매 10수마다 간략한 진척 상황 출력 (버퍼링 우회)
        if move_count % 10 == 0:
            elapsed_move = time.time() - t0
            print(f"   [Game {game_idx:03d}] Move {move_count:03d} processed (elapsed: {elapsed_move:.1f}s)", flush=True)
            
    winner = game.winner
    if winner is None:
        winner = game.check_winner()
        
    if game.game_over:
        if game.consecutive_passes >= 2:
            reason = "TERRITORY"
        else:
            reason = "CAPTURE"
    else:
        reason = "MAX_MOVES"
        
    elapsed_game = time.time() - t0
    # 게임 1판 완료 즉시 상세 로깅
    print(f"-> [Game {game_idx:03d} Finished] Winner: {winner} | Reason: {reason} | Moves: {move_count} | Time: {elapsed_game:.1f}s", flush=True)
        
    # JSON 파일 포맷 구성
    game_data = {
        "winner": winner,
        "win_reason": reason,
        "moves": moves_history
    }
    
    os.makedirs("data/games", exist_ok=True)
    filename = f"data/games/game_{game_idx:03d}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(game_data, f, indent=2)
        
    return {
        "game_idx": game_idx,
        "winner": winner,
        "win_reason": reason,
        "moves": moves_history,
        "moves_count": move_count
    }

def run_analysis(results):
    num_games = len(results)
    if num_games == 0:
        return
        
    blue_wins = sum(1 for r in results if r["winner"] == 1)
    orange_wins = sum(1 for r in results if r["winner"] == 2)
    draws = sum(1 for r in results if r["winner"] not in (1, 2) or r["winner"] is None)
    
    capture_count = sum(1 for r in results if r["win_reason"] == "CAPTURE")
    territory_count = sum(1 for r in results if r["win_reason"] == "TERRITORY")
    max_moves_count = sum(1 for r in results if r["win_reason"] == "MAX_MOVES")
    
    avg_moves = sum(r["moves_count"] for r in results) / num_games
    
    # 첫 수 TOP 10
    first_moves = {}
    for r in results:
        if r["moves"]:
            first_move = tuple(r["moves"][0])
            first_moves[first_move] = first_moves.get(first_move, 0) + 1
            
    sorted_first_moves = sorted(first_moves.items(), key=lambda x: x[1], reverse=True)
    
    # 중복 게임 비율
    unique_games = set()
    for r in results:
        serialized = json.dumps(r["moves"])
        unique_games.add(serialized)
        
    duplicate_ratio = (num_games - len(unique_games)) / num_games * 100.0
    
    print("\n================ SELF-PLAY QUALITY ANALYSIS ================")
    print(f"Total Games Analyzed: {num_games}")
    print("-" * 55)
    print(f"1. 승률")
    print(f"   - BLUE Wins   : {blue_wins} ({blue_wins/num_games*100:.2f}%)")
    print(f"   - ORANGE Wins : {orange_wins} ({orange_wins/num_games*100:.2f}%)")
    print(f"   - Draws       : {draws} ({draws/num_games*100:.2f}%)")
    print("-" * 55)
    print(f"2. Capture 비율  : {capture_count} ({capture_count/num_games*100:.2f}%)")
    print(f"3. Territory 비율: {territory_count} ({territory_count/num_games*100:.2f}%)")
    print(f"   (Max Moves 비율: {max_moves_count} ({max_moves_count/num_games*100:.2f}%))")
    print("-" * 55)
    print(f"4. 평균 수순     : {avg_moves:.2f} 수")
    print("-" * 55)
    print(f"5. 첫 수 TOP 10 (좌표: 빈도수):")
    for idx, (move, freq) in enumerate(sorted_first_moves[:10]):
        print(f"   #{idx+1:02d} | Move {move} : {freq}회 ({freq/num_games*100:.2f}%)")
    print("-" * 55)
    print(f"6. 중복 게임 비율: {duplicate_ratio:.2f}% (고유 게임: {len(unique_games)}/{num_games})")
    print("============================================================\n")

def run_self_play_simulation(num_games=500, depth=3):
    print("=== STARTING PARALLEL SELF-PLAY SIMULATION (500 GAMES) ===", flush=True)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    print(f"Cores Detected: {num_cores} | Using Processes: {num_processes}", flush=True)
    
    start_time = time.time()
    
    tasks = [(i, depth) for i in range(1, num_games + 1)]
    results = []
    completed = 0
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for res in pool.imap_unordered(play_single_game_self_play, tasks):
            results.append(res)
            completed += 1
            if completed % 10 == 0 or completed == num_games:
                print(f"[Progress] {completed}/{num_games} games completed ({completed/num_games*100:.1f}%)", flush=True)
                
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).", flush=True)
    
    # 결과 분석 실행
    run_analysis(results)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_self_play_simulation(num_games=10, depth=3)
