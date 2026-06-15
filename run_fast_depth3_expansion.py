import io
import os
import sys
import time
import random
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from contextlib import redirect_stdout

# Add project path to sys.path
sys.path.append(r"C:\Users\User\source\repos\greatkingdomAI")

from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from ai.minimax import alphabeta, find_best_move, clear_transposition_table, reset_stats, get_legal_moves
from model_v2 import PolicyNetworkV2
from ai.hybrid import find_hybrid_move, board_to_tensor, get_move_idx, get_move_from_idx
from dataset import GreatKingdomDataset

_worker_model = None

def init_worker(model_path):
    global _worker_model
    device = torch.device("cpu")
    _worker_model = PolicyNetworkV2().to(device)
    if os.path.exists(model_path):
        _worker_model.load_state_dict(torch.load(model_path, map_location=device))
    _worker_model.eval()

# ----------------- Policy Guided Depth-3 Move Search -----------------
def find_policy_guided_depth3_move(game_state, policy_model, device, k=8):
    if game_state.game_over:
        return "pass"
        
    curr_player = game_state.current_player
    legal_moves = get_legal_moves(game_state)
    if not legal_moves:
        return "pass"
        
    state_np = board_to_tensor(game_state.board, curr_player)
    state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = policy_model(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    move_probs = []
    for m in legal_moves:
        idx = get_move_idx(m)
        prob = probs[idx]
        move_probs.append((m, prob))
        
    move_probs.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [m for m, _ in move_probs[:k]]
    
    from ai.evaluation import clear_future_liberty_cache
    clear_future_liberty_cache()
    from engine.safe_groups import clear_empty_regions_cache
    clear_empty_regions_cache()
    from ai.minimax import copy_game_state
    
    target_player = curr_player
    alpha = -float("inf")
    beta = float("inf")
    move_scores = []
    
    for move in top_candidates:
        next_state = copy_game_state(game_state)
        try:
            if move == "pass":
                next_state.play_pass()
            else:
                next_state.play_move(move[0], move[1])
        except ValueError:
            continue
            
        score = alphabeta(next_state, depth=2, alpha=alpha, beta=beta, maximizing_player=False, target_player=target_player)
        move_scores.append((move, score))
        alpha = max(alpha, score)
        
    if move_scores:
        move_scores.sort(key=lambda x: x[1], reverse=True)
        best_score = move_scores[0][1]
        best_candidates = [move for move, score in move_scores if abs(score - best_score) < 1e-7]
        return random.choice(best_candidates)
        
    return "pass"

# ----------------- Selfplay worker function -----------------
def play_one_selfplay_fast_depth3(args):
    game_idx, model_path = args
    global _worker_model
    
    if _worker_model is None:
        init_worker(model_path)
        
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    game.is_copy = False
    
    device = torch.device("cpu")
    history = []
    move_count = 0
    max_moves = 150
    
    while not game.game_over and move_count < max_moves:
        current_player = game.current_player
        state_tensor = board_to_tensor(game.board, current_player)
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            move = find_policy_guided_depth3_move(game, _worker_model, device, k=8)
            
        action_coords = np.array([-1, -1], dtype=np.int8) if move == "pass" else np.array(move, dtype=np.int8)
        history.append((state_tensor, action_coords, current_player))
        
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                game.play_pass()
                history[-1] = (state_tensor, np.array([-1, -1], dtype=np.int8), current_player)
                
        move_count += 1
        
    winner = game.winner if game.winner is not None else game.check_winner()
    
    if game.game_over:
        termination = "CAPTURE" if game.consecutive_passes < 2 else "PASS"
    else:
        termination = "MAX_MOVES"
        
    return {
        "game_idx": game_idx,
        "winner": winner,
        "moves": move_count,
        "termination": termination,
        "history": history
    }

def run_selfplay_fast_depth3(num_games, model_path, output_path):
    print(f"\n=== Step 1: Generating {num_games} Self-Play Games with Fast Depth3 (K=8) ===", flush=True)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    print(f"  Using {num_processes} parallel processes.", flush=True)
    
    start_time = time.time()
    
    all_states = []
    all_actions = []
    all_players = []
    all_results = []
    all_game_ids = []
    all_terminations = [None] * num_games
    
    completed = 0
    total_moves = 0
    moves_counts = []
    move_sequences = []
    first_moves = []
    
    pool_args = [(i, model_path) for i in range(1, num_games + 1)]
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
        for res in pool.imap_unordered(play_one_selfplay_fast_depth3, pool_args):
            completed += 1
            game_idx = res["game_idx"]
            winner = res["winner"]
            moves = res["moves"]
            termination = res["termination"]
            game_history = res["history"]
            
            total_moves += moves
            moves_counts.append(moves)
            all_terminations[game_idx - 1] = termination
            
            seq = []
            for _, act, _ in game_history:
                seq.append(tuple(act))
            move_sequences.append(tuple(seq))
            
            if len(seq) > 0:
                first_moves.append(seq[0])
                
            for state_tensor, action_coords, player_id in game_history:
                all_states.append(state_tensor)
                all_actions.append(action_coords)
                all_players.append(player_id)
                all_results.append(winner)
                all_game_ids.append(game_idx)
                
            if completed % 50 == 0 or completed == num_games:
                elapsed = time.time() - start_time
                print(f"    [Generation Progress {completed:04d}/{num_games:04d}] Elapsed: {elapsed:.1f}s | Avg Moves: {total_moves/completed:.1f}", flush=True)
                
    # Save npz
    states_arr = np.array(all_states, dtype=np.int8)
    actions_arr = np.array(all_actions, dtype=np.int8)
    players_arr = np.array(all_players, dtype=np.int8)
    results_arr = np.array(all_results, dtype=np.int8)
    game_ids_arr = np.array(all_game_ids, dtype=np.int32)
    terminations_arr = np.array(all_terminations, dtype='U10')
    
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
    
    unique_seqs = set(move_sequences)
    unique_ratio = len(unique_seqs) / num_games
    duplicate_ratio = 1.0 - unique_ratio
    avg_moves = np.mean(moves_counts)
    
    print(f"  Dataset generated at {output_path} (Size: {file_size_mb:.2f} MB)", flush=True)
    print(f"  Uniqueness Ratio: {unique_ratio*100:.2f}% | Duplication Ratio: {duplicate_ratio*100:.2f}%", flush=True)
    
    return {
        "games": num_games,
        "samples": len(all_states),
        "duration": total_elapsed,
        "unique_ratio": unique_ratio,
        "duplicate_ratio": duplicate_ratio,
        "avg_moves": avg_moves
    }

# ----------------- Phase 2: Model Training -----------------
def get_topk_correct(outputs, targets, k):
    _, topk_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    return correct.any(dim=1).sum().item()

def run_train_model(npz_path, model_path, epochs=20):
    print("\n=== Step 2: Training Policy Network V2 on GPU ===", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training device: {device}", flush=True)
    
    train_dataset = GreatKingdomDataset(npz_path, mode="train", split_ratio=0.9)
    val_dataset = GreatKingdomDataset(npz_path, mode="val", split_ratio=0.9)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    print(f"  Train samples     : {len(train_dataset)}", flush=True)
    print(f"  Validation samples: {len(val_dataset)}", flush=True)
    
    model = PolicyNetworkV2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )
    
    history = []
    best_val_loss = float("inf")
    best_val_acc1 = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for states, targets in train_loader:
            states = states.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * states.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct1 = 0
        val_correct3 = 0
        val_correct5 = 0
        
        with torch.no_grad():
            for states, targets in val_loader:
                states = states.to(device)
                targets = targets.to(device)
                
                outputs = model(states)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * states.size(0)
                
                val_correct1 += get_topk_correct(outputs, targets, k=1)
                val_correct3 += get_topk_correct(outputs, targets, k=3)
                val_correct5 += get_topk_correct(outputs, targets, k=5)
                
        val_loss /= len(val_dataset)
        val_acc1 = (val_correct1 / len(val_dataset)) * 100
        val_acc3 = (val_correct3 / len(val_dataset)) * 100
        val_acc5 = (val_correct5 / len(val_dataset)) * 100
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc3": val_acc3,
            "val_acc5": val_acc5
        })
        
        print(f"    Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc1: {val_acc1:.2f}% | Val Acc3: {val_acc3:.2f}% | Val Acc5: {val_acc5:.2f}%", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc1 = val_acc1
            torch.save(model.state_dict(), model_path)
            
    total_elapsed = time.time() - start_time
    print(f"  Training completed (Saved to {model_path}) | Took: {total_elapsed:.1f}s", flush=True)
    return history

# ----------------- Phase 3: Match Verification -----------------
_model_new = None
_model_temp08 = None

def init_match_worker(model_new_path, model_temp08_path):
    global _model_new, _model_temp08
    device = torch.device("cpu")
    
    _model_new = PolicyNetworkV2().to(device)
    if os.path.exists(model_new_path):
        _model_new.load_state_dict(torch.load(model_new_path, map_location=device))
    _model_new.eval()
    
    _model_temp08 = PolicyNetworkV2().to(device)
    if os.path.exists(model_temp08_path):
        _model_temp08.load_state_dict(torch.load(model_temp08_path, map_location=device))
    _model_temp08.eval()

def get_pure_policy_move(game, model, device):
    state_np = board_to_tensor(game.board, game.current_player)
    state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    legal_moves = get_legal_moves(game)
    if not legal_moves:
        return "pass"
    legal_indices = [get_move_idx(m) for m in legal_moves]
    
    legal_probs = probs[legal_indices]
    if np.sum(legal_probs) > 0:
        legal_probs /= np.sum(legal_probs)
        best_idx = np.argmax(legal_probs)
        chosen_idx = legal_indices[best_idx]
    else:
        chosen_idx = legal_indices[0]
    return get_move_from_idx(chosen_idx)

def play_new_vs_temp08(args):
    game_idx, new_color = args
    global _model_new, _model_temp08
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    device = torch.device("cpu")
    
    move_count = 0
    max_moves = 150
    game_start = time.time()
    
    while not game.game_over and move_count < max_moves:
        curr_player = game.current_player
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if curr_player == new_color:
                move = get_pure_policy_move(game, _model_new, device)
            else:
                move = get_pure_policy_move(game, _model_temp08, device)
                
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                game.play_pass()
                
        move_count += 1
        
    winner = game.winner if game.winner is not None else game.check_winner()
    return {"winner": winner, "moves": move_count, "duration": time.time() - game_start, "new_color": new_color}

def play_new_vs_hybrid(args):
    game_idx, new_color = args
    global _model_new
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    device = torch.device("cpu")
    
    move_count = 0
    max_moves = 150
    game_start = time.time()
    
    while not game.game_over and move_count < max_moves:
        curr_player = game.current_player
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if curr_player == new_color:
                move = get_pure_policy_move(game, _model_new, device)
            else:
                move = find_hybrid_move(game, _model_new, device, temperature=None)
                
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                game.play_pass()
                
        move_count += 1
        
    winner = game.winner if game.winner is not None else game.check_winner()
    return {"winner": winner, "moves": move_count, "duration": time.time() - game_start, "new_color": new_color}

def play_new_vs_depth3(args):
    game_idx, new_color = args
    global _model_new
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    device = torch.device("cpu")
    
    move_count = 0
    max_moves = 150
    game_start = time.time()
    
    while not game.game_over and move_count < max_moves:
        curr_player = game.current_player
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if curr_player == new_color:
                move = get_pure_policy_move(game, _model_new, device)
            else:
                f = io.StringIO()
                with redirect_stdout(f):
                    move = find_best_move(game, depth=3)
                    
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                game.play_pass()
                
        move_count += 1
        
    winner = game.winner if game.winner is not None else game.check_winner()
    return {"winner": winner, "moves": move_count, "duration": time.time() - game_start, "new_color": new_color}

def main():
    print("=================================================================")
    print("GREAT KINGDOM AI - FAST DEPTH3 1,000 GAMES EXPANSION & EVALUATION")
    print("=================================================================")
    
    model_temp08_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_new_temp.pth"
    model_new_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_fast_depth3_1000.pth"
    npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_fast_depth3_1000.npz"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\fast_depth3_final_report.md"
    artifact_report_path = r"C:\Users\User\.gemini\antigravity-cli\brain\505d76c0-2e72-46fd-bbbf-aa02a413b645\fast_depth3_final_report.md"
    
    # 1. Generate 1,000 Games of Self-Play
    gen_stats = run_selfplay_fast_depth3(1000, model_temp08_path, npz_path)
    
    # 2. Train Policy Network V2 with the 1,000-game dataset
    history = run_train_model(npz_path, model_new_path, epochs=20)
    
    # 3. Matchup Evaluators (50 games each)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    
    tasks = []
    for i in range(1, 51):
        color = BLUE if i <= 25 else ORANGE
        tasks.append((i, color))
        
    print("\n=== Step 3: Running Matchup Verification Battles (50 games each) ===", flush=True)
    
    # Matchup 1: New vs Temp=0.8 Model
    print("  1/3: New Model vs Temp=0.8 Model...", flush=True)
    new_v_temp08_wins = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_match_worker, initargs=(model_new_path, model_temp08_path)) as pool:
        for res in pool.imap_unordered(play_new_vs_temp08, tasks):
            if res["winner"] == res["new_color"]:
                new_v_temp08_wins += 1
    new_v_temp08_rate = (new_v_temp08_wins / 50) * 100
    print(f"  => Win Rate vs Temp0.8: {new_v_temp08_rate:.1f}% ({new_v_temp08_wins}/50)", flush=True)
    
    # Matchup 2: New vs Hybrid Teacher
    print("  2/3: New Model vs Hybrid Teacher...", flush=True)
    new_v_hybrid_wins = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_match_worker, initargs=(model_new_path, model_temp08_path)) as pool:
        for res in pool.imap_unordered(play_new_vs_hybrid, tasks):
            if res["winner"] == res["new_color"]:
                new_v_hybrid_wins += 1
    new_v_hybrid_rate = (new_v_hybrid_wins / 50) * 100
    print(f"  => Win Rate vs Hybrid Teacher: {new_v_hybrid_rate:.1f}% ({new_v_hybrid_wins}/50)", flush=True)
    
    # Matchup 3: New vs Depth 3 Minimax
    print("  3/3: New Model vs Depth 3...", flush=True)
    new_v_d3_wins = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_match_worker, initargs=(model_new_path, model_temp08_path)) as pool:
        for res in pool.imap_unordered(play_new_vs_depth3, tasks):
            if res["winner"] == res["new_color"]:
                new_v_d3_wins += 1
    new_v_d3_rate = (new_v_d3_wins / 50) * 100
    print(f"  => Win Rate vs Depth3: {new_v_d3_rate:.1f}% ({new_v_d3_wins}/50)", flush=True)
    
    # Evaluate success criteria
    if new_v_d3_rate >= 50.0:
        d3_verdict = "대형 성공 (50% 이상)"
    elif new_v_d3_rate >= 40.0:
        d3_verdict = "매우 성공 (40% 이상)"
    elif new_v_d3_rate >= 35.0:
        d3_verdict = "성공 (35% 이상)"
    else:
        d3_verdict = "성공 기준 미달 (35% 미만)"
        
    overall_success = new_v_d3_rate >= 35.0
    final_verdict_str = "SUCCESS (일반화 성능 향상 입증)" if overall_success else "FAIL (성공 기준 미달)"
    
    # Write Final report
    md = []
    md.append("# Great Kingdom AI - Fast Depth3 1,000 Games Final Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **학습 조건**: Fast Depth3 (K=8) 1,000판 데이터셋 ({gen_stats['samples']:,} 샘플, {gen_stats['unique_ratio']*100:.2f}% Diversity)\n")
    
    md.append("## 1. 데이터셋 다양성 결과")
    md.append(f"* **고유 기보 비율**: **{gen_stats['unique_ratio']*100:.2f}%**")
    md.append(f"* **중복 기보 비율**: **{gen_stats['duplicate_ratio']*100:.2f}%**")
    md.append(f"* **총 게임 수**: {gen_stats['games']} 판")
    md.append(f"* **총 샘플 수**: {gen_stats['samples']:,} 샘플")
    md.append(f"* **평균 게임 길이**: {gen_stats['avg_moves']:.1f} 수\n")
    
    md.append("## 2. 정책 네트워크(Policy Network) 학습 결과 기록")
    md.append(f"* **최종 Epoch**: {len(history)}")
    md.append(f"* **최종 Train Loss**: {history[-1]['train_loss']:.4f}")
    md.append(f"* **최종 Validation Loss**: {history[-1]['val_loss']:.4f}")
    md.append(f"* **최종 Validation Accuracy (Top-1)**: **{history[-1]['val_acc1']:.2f}%**")
    md.append(f"* **최종 Validation Accuracy (Top-3)**: **{history[-1]['val_acc3']:.2f}%**")
    md.append(f"* **최종 Validation Accuracy (Top-5)**: **{history[-1]['val_acc5']:.2f}%**\n")
    
    md.append("### 에포크별 Loss & Accuracy 추이 (Loss Curve & Acc)")
    md.append("| Epoch | Train Loss | Val Loss | Val Acc1 | Val Acc3 | Val Acc5 |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for h in history:
        md.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | {h['val_acc1']:.2f}% | {h['val_acc3']:.2f}% | {h['val_acc5']:.2f}% |")
    md.append("")
    
    md.append("## 3. 실전 대국 검증 결과 (Matchup Statistics)")
    md.append("| 평가 대상 (Matchup) | 총 판수 | 새 모델 (Fast Depth3 1000판) 승률 | 비고 |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| **새 모델 vs 기존 Temp0.8 모델** | 50판 | **{new_v_temp08_rate:.1f}%** ({new_v_temp08_wins}승) | 기존 최고 실전 모델과의 비교 |")
    md.append(f"| **새 모델 vs Hybrid Teacher (결정론)** | 50판 | **{new_v_hybrid_rate:.1f}%** ({new_v_hybrid_wins}승) | 하이브리드 엔진 모방 지표 |")
    md.append(f"| **새 모델 vs Depth 3 Minimax** | 50판 | **{new_v_d3_rate:.1f}%** ({new_v_d3_wins}승) | **성공 판정: {d3_verdict}** |")
    md.append("")
    
    md.append("## 4. 핵심 모델별 정량 비교 분석")
    md.append("| 지표 (Metrics) | 기존 최고 모델 (Temp0.8, K=5) | Multi-Teacher 모델 | **신규 Fast Depth3 1,000판 모델** |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **학습 데이터 크기 (샘플 수)** | 50,000 | 42,939 | **{gen_stats['samples']:,}** |")
    md.append(f"| **데이터셋 고유 기보 비율** | 10.30% | 36.00% | **{gen_stats['unique_ratio']*100:.2f}%** |")
    md.append(f"| **Depth 3 상대 승률** | 30.0% | 36.0% (과거값) | **{new_v_d3_rate:.1f}%** |")
    md.append("")
    
    md.append("## 5. 최종 결론: 좋은 Teacher가 많은 데이터보다 중요한가?")
    md.append(f"### 최종 판정: **{final_verdict_str}**")
    
    if overall_success:
        md.append(f"- **결론**: **좋은 Teacher와 충분한 데이터 규모의 조합이 일반화 실력 극대화를 이끌어냈습니다.**")
        md.append(f"- 고유 기보 비율 {gen_stats['unique_ratio']*100:.2f}%를 자랑하는 Fast Depth3 교사의 1,000판 데이터셋을 통해 Depth3 상대 승률 **{new_v_d3_rate:.1f}%**를 달성하여 성공 기준(35% 이상)을 크게 넘어섰습니다.")
        md.append("- 이는 기존 다중 교사 혼합이나 결정론적 데이터의 한계를 넘어, 고품질 지도 학습만으로도 강력한 일반화 정책 학습이 가능함을 규명한 성과입니다.")
    else:
        md.append(f"- **결론**: 1,000판(약 {gen_stats['samples']:,} 샘플) 데이터 규모의 확충에도 불구하고 Depth3 상대 승률이 35% 미만에 머물렀습니다.")
        md.append("- 이는 단일 Fast Depth3 교사의 포석 무작위성만으로는 다양한 패턴에 대한 강인한 일반화 대응을 구축하기에 학습 신호의 다양성 한계나 하이퍼파라미터 튜닝이 추가로 요구됨을 시사합니다.")
        
    md_content = "\n".join(md)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Report successfully saved to: {report_path}", flush=True)
    
    os.makedirs(os.path.dirname(artifact_report_path), exist_ok=True)
    with open(artifact_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Artifact report successfully saved to: {artifact_report_path}", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
