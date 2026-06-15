import torch
import numpy as np
import time
from engine.game_state import GameState
from ai.minimax import get_legal_moves, copy_game_state
from ai.hybrid import board_to_tensor

VALUE_STATS = {"nodes_visited": 0}

def get_leaf_states_dfs(state, depth, path_states, leaf_map):
    """
    Recursively traverse the game tree to collect all leaf states.
    Uses Zobrist hash of state to avoid duplicate evaluations.
    """
    if depth == 0 or state.game_over:
        hash_key = state.hash_val
        if hash_key not in leaf_map:
            leaf_map[hash_key] = state
        return
        
    legal_moves = get_legal_moves(state)
    for move in legal_moves:
        next_state = copy_game_state(state)
        try:
            if move == "pass":
                next_state.play_pass()
            else:
                next_state.play_move(move[0], move[1])
        except ValueError:
            continue
            
        get_leaf_states_dfs(next_state, depth - 1, path_states, leaf_map)

def batch_evaluate_states(leaf_map, model, device):
    """
    Convert all leaf states to tensors and perform batched GPU inference.
    Returns a dict mapping hash_key -> target_player_score
    """
    if not leaf_map:
        return {}
        
    hash_keys = list(leaf_map.keys())
    states_list = list(leaf_map.values())
    
    tensors = []
    for s in states_list:
        # Model expects input channel representation from current_player's perspective
        tensor_np = board_to_tensor(s.board, s.current_player)
        tensors.append(tensor_np)
        
    batch_tensor = torch.tensor(np.array(tensors), dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        values = model(batch_tensor).cpu().numpy()
        
    cache = {}
    for i, hash_key in enumerate(hash_keys):
        s = states_list[i]
        val = values[i] # scalar in [-1, 1] from current_player's perspective
        
        # In minimax, we store the score relative to the current player
        # so when looking up, we multiply by appropriate sign
        cache[hash_key] = val
        
    return cache

def alphabeta_value(state, depth, alpha, beta, maximizing_player, target_player, value_cache):
    VALUE_STATS["nodes_visited"] += 1
    
    if depth == 0 or state.game_over:
        if state.game_over:
            if state.winner == target_player:
                return 100000.0 + depth
            elif state.winner is not None:
                return -100000.0 - depth
            else:
                return 0.0
        else:
            hash_key = state.hash_val
            val = value_cache.get(hash_key, 0.0)
            
            # Convert current player's perspective value to target player's perspective
            if state.current_player != target_player:
                val = -val
            return val
            
    legal_moves = get_legal_moves(state)
    # We sort moves to optimize pruning
    from ai.minimax import order_moves
    legal_moves = order_moves(state, legal_moves, depth, maximizing_player, target_player)
    
    if maximizing_player:
        max_eval = -float("inf")
        for move in legal_moves:
            next_state = copy_game_state(state)
            try:
                if move == "pass":
                    next_state.play_pass()
                else:
                    next_state.play_move(move[0], move[1])
            except ValueError:
                continue
                
            eval_val = alphabeta_value(next_state, depth - 1, alpha, beta, False, target_player, value_cache)
            max_eval = max(max_eval, eval_val)
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in legal_moves:
            next_state = copy_game_state(state)
            try:
                if move == "pass":
                    next_state.play_pass()
                else:
                    next_state.play_move(move[0], move[1])
            except ValueError:
                continue
                
            eval_val = alphabeta_value(next_state, depth - 1, alpha, beta, True, target_player, value_cache)
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval

def find_value_minimax_move(game_state, value_model, device="cpu", depth=2):
    """
    Depth2 Minimax Search with batched Value Network leaf evaluations.
    """
    if game_state.game_over:
        return "pass"
        
    target_player = game_state.current_player
    
    # 1. Collect all reachable leaf states
    leaf_map = {}
    get_leaf_states_dfs(game_state, depth, [], leaf_map)
    
    # 2. Batch evaluate all leaves on GPU
    value_cache = batch_evaluate_states(leaf_map, value_model, device)
    
    # 3. Minimax Alpha-Beta search using cached values
    legal_moves = get_legal_moves(game_state)
    move_scores = []
    
    alpha = -float("inf")
    beta = float("inf")
    
    for move in legal_moves:
        next_state = copy_game_state(game_state)
        try:
            if move == "pass":
                next_state.play_pass()
            else:
                next_state.play_move(move[0], move[1])
        except ValueError:
            continue
            
        score = alphabeta_value(next_state, depth - 1, alpha, beta, False, target_player, value_cache)
        move_scores.append((move, score))
        alpha = max(alpha, score)
        
    if not move_scores:
        return "pass"
        
    move_scores.sort(key=lambda x: x[1], reverse=True)
    best_score = move_scores[0][1]
    best_moves = [m for m, s in move_scores if abs(s - best_score) < 1e-5]
    
    import random
    return random.choice(best_moves)
