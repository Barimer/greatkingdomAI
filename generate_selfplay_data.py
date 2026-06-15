import sys
import os
import time
import argparse
import multiprocessing
import numpy as np

# Add project path to sys.path
sys.path.append(r"C:\Users\User\source\repos\greatkingdomAI")

def board_to_tensor(board, current_player):
    """
    board: Board object
    current_player: 1 (BLUE) or 2 (ORANGE)
    Returns a numpy array of shape (4, 9, 9) representing:
      - Channel 0: Current player's stones
      - Channel 1: Opponent's stones
      - Channel 2: Neutral castles
      - Channel 3: Empty cells
    """
    opponent = 2 if current_player == 1 else 1
    state = np.zeros((4, 9, 9), dtype=np.int8)  # int8 saves disk space compared to float32
    
    for r in range(9):
        for c in range(9):
            val = board.get(r, c)
            if val == current_player:
                state[0, r, c] = 1
            elif val == opponent:
                state[1, r, c] = 1
            elif val == 3:  # NEUTRAL
                state[2, r, c] = 1
            else:  # EMPTY
                state[3, r, c] = 1
                
    return state

def play_one_selfplay_game(args):
    game_idx, depth = args
    from engine.game_state import GameState
    from ai.minimax import find_best_move, clear_transposition_table, reset_stats
    import random
    import io
    from contextlib import redirect_stdout
    
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    game.is_copy = True
    
    history = []
    move_count = 0
    max_moves = 150
    
    while not game.game_over and move_count < max_moves:
        current_player = game.current_player
        
        # Save state tensor BEFORE the move
        state_tensor = board_to_tensor(game.board, current_player)
        
        # Determine move
        if move_count == 0:
            from ai.minimax import get_legal_moves
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            f = io.StringIO()
            with redirect_stdout(f):
                move = find_best_move(game, depth=depth)
                
        # Record history entry: (state, action, player)
        action_coords = np.array([-1, -1], dtype=np.int8) if move == "pass" else np.array(move, dtype=np.int8)
        history.append((state_tensor, action_coords, current_player))
        
        # Play move
        if move == "pass":
            game.play_pass()
        else:
            game.play_move(move[0], move[1])
            
        move_count += 1
        
    winner = game.winner
    if winner is None:
        winner = game.check_winner()
        
    if game.game_over:
        termination = "CAPTURE" if game.consecutive_passes < 2 else "PASS"
    else:
        termination = "MAX_MOVES"
        
    # Return game stats and the history
    return {
        "game_idx": game_idx,
        "winner": winner,
        "moves": move_count,
        "termination": termination,
        "history": history
    }

def main():
    parser = argparse.ArgumentParser(description="Great Kingdom AI - Self Play Data Generator")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to simulate")
    parser.add_argument("--depth", type=int, default=2, help="Minimax search depth")
    parser.add_argument("--output", type=str, default=r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_dataset_v1.npz", help="Output .npz file path")
    args = parser.parse_args()
    
    num_games = args.num_games
    depth = args.depth
    output_path = args.output
    
    print("=== GREAT KINGDOM AI - SELF PLAY DATA GENERATOR ===")
    print(f"Games to simulate: {num_games}")
    print(f"Minimax Depth    : {depth}")
    print(f"Output File      : {output_path}")
    print("-" * 50)
    
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    print(f"Detected Cores   : {num_cores} | Active Processes: {num_processes}")
    
    start_time = time.time()
    
    all_states = []
    all_actions = []
    all_players = []
    all_results = []
    all_game_ids = []
    all_terminations = [None] * num_games
    
    completed = 0
    total_moves = 0
    
    pool_args = [(i, depth) for i in range(1, num_games + 1)]
    
    # Run games in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        for res in pool.imap_unordered(play_one_selfplay_game, pool_args):
            completed += 1
            game_idx = res["game_idx"]
            winner = res["winner"]
            moves = res["moves"]
            termination = res["termination"]
            history = res["history"]
            
            total_moves += moves
            all_terminations[game_idx - 1] = termination
            
            # Pack history into global lists
            for state_tensor, action_coords, player_id in history:
                all_states.append(state_tensor)
                all_actions.append(action_coords)
                all_players.append(player_id)
                all_results.append(winner)
                all_game_ids.append(game_idx)
                
            elapsed = time.time() - start_time
            print(f"[{completed:03d}/{num_games:03d}] Game #{game_idx:03d} finished | Moves: {moves:3d} | Winner: {winner} | Cumulative Moves: {total_moves} | Elapsed: {elapsed:.1f}s")
            sys.stdout.flush()
            
    # Convert lists to numpy arrays
    print("\nPacking data into numpy arrays...", flush=True)
    states_arr = np.array(all_states, dtype=np.int8)
    actions_arr = np.array(all_actions, dtype=np.int8)
    players_arr = np.array(all_players, dtype=np.int8)
    results_arr = np.array(all_results, dtype=np.int8)
    game_ids_arr = np.array(all_game_ids, dtype=np.int32)
    terminations_arr = np.array(all_terminations, dtype='U10')
    
    # Save as compressed .npz
    print(f"Saving to {output_path}...", flush=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states_arr,
        actions=actions_arr,
        players=players_arr,
        results=results_arr,
        game_ids=game_ids_arr,
        terminations=terminations_arr
    )
    
    total_elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print("=" * 50)
    print("Self Play Data Generation Completed!")
    print(f"Total Games Played : {num_games}")
    print(f"Total Moves Saved  : {len(all_states)}")
    print(f"Total Time Elapsed : {total_elapsed:.1f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Average Game Length: {total_moves/num_games:.1f} moves")
    print(f"Average Time/Game  : {total_elapsed/num_games:.1f} seconds")
    print(f"Dataset File Size  : {file_size_mb:.2f} MB")
    print("=" * 50)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
