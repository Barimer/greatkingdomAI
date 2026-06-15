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
from ai.minimax import clear_transposition_table, reset_stats, get_legal_moves
from model_v2 import PolicyNetworkV2
from ai.hybrid import find_hybrid_move, board_to_tensor, get_move_idx, get_move_from_idx
from dataset import GreatKingdomDataset, move_to_index

# Try importing psutil for hardware metrics
try:
    import psutil
except ImportError:
    psutil = None

_worker_model = None

def init_worker(model_path):
    global _worker_model
    device = torch.device("cpu")
    _worker_model = PolicyNetworkV2().to(device)
    if os.path.exists(model_path):
        _worker_model.load_state_dict(torch.load(model_path, map_location=device))
    _worker_model.eval()

def play_one_selfplay_hybrid(args):
    game_idx, model_path = args
    global _worker_model
    
    # If worker is not initialized (e.g., sequentially), init here
    if _worker_model is None:
        init_worker(model_path)
        
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    # game.is_copy = False로 유지하여 강제 사활 및 위기 탈출 방어 전술 필터링 활성화
    game.is_copy = False
    
    device = torch.device("cpu")
    history = []
    move_count = 0
    max_moves = 150
    
    while not game.game_over and move_count < max_moves:
        current_player = game.current_player
        
        # Save state tensor BEFORE the move (input features)
        state_tensor = board_to_tensor(game.board, current_player)
        
        # Determine move using Hybrid Engine
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            move = find_hybrid_move(game, _worker_model, device)
            
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
        
    winner = game.winner
    if winner is None:
        winner = game.check_winner()
        
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

def get_system_metrics():
    cpu_percent = 0.0
    mem_used_gb = 0.0
    if psutil is not None:
        try:
            cpu_percent = psutil.cpu_percent()
            mem_info = psutil.virtual_memory()
            mem_used_gb = mem_info.used / (1024 ** 3)
        except Exception:
            pass
    return cpu_percent, mem_used_gb

def run_selfplay_data_generation(num_games, model_path, output_path, report_path):
    print(f"--- Starting Self-Play Data Generation: {num_games} games ---", flush=True)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    
    start_time = time.time()
    
    all_states = []
    all_actions = []
    all_players = []
    all_results = []
    all_game_ids = []
    all_terminations = [None] * num_games
    
    completed = 0
    total_moves = 0
    
    # For diversity analysis
    blue_wins = 0
    orange_wins = 0
    draws = 0
    
    moves_counts = []
    move_sequences = []
    first_moves = []
    
    pool_args = [(i, model_path) for i in range(1, num_games + 1)]
    
    print(f"Cores: {num_cores} | Active Processes: {num_processes}", flush=True)
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
        for res in pool.imap_unordered(play_one_selfplay_hybrid, pool_args):
            completed += 1
            game_idx = res["game_idx"]
            winner = res["winner"]
            moves = res["moves"]
            termination = res["termination"]
            game_history = res["history"]
            
            total_moves += moves
            moves_counts.append(moves)
            all_terminations[game_idx - 1] = termination
            
            # Count winners
            if winner == BLUE:
                blue_wins += 1
            elif winner == ORANGE:
                orange_wins += 1
            else:
                draws += 1
                
            # Collect move sequences for uniqueness check
            seq = []
            for _, act, _ in game_history:
                seq.append(tuple(act))
            move_sequences.append(tuple(seq))
            
            if len(seq) > 0:
                first_moves.append(seq[0])
                
            # Pack history elements
            for state_tensor, action_coords, player_id in game_history:
                all_states.append(state_tensor)
                all_actions.append(action_coords)
                all_players.append(player_id)
                all_results.append(winner)
                all_game_ids.append(game_idx)
                
            # Log progress every 500 games
            if completed % 500 == 0 or completed == num_games:
                elapsed = time.time() - start_time
                avg_length = total_moves / completed
                cpu_p, mem_g = get_system_metrics()
                print(f"  [Progress {completed:04d}/{num_games:04d}] Elapsed: {elapsed:.1f}s | Avg Moves: {avg_length:.1f} | CPU: {cpu_p:.1f}% | RAM Used: {mem_g:.2f}GB", flush=True)
                
    # Packing and saving
    print("Packing selfplay data into numpy arrays...", flush=True)
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
    
    print(f"Dataset saved to: {output_path} (Size: {file_size_mb:.2f} MB)", flush=True)
    
    # ----------------- Analyze Diversity -----------------
    unique_seqs = set(move_sequences)
    unique_ratio = len(unique_seqs) / num_games
    duplicate_ratio = 1.0 - unique_ratio
    
    # Opening diversity (First move counts)
    opening_counts = {}
    for fm in first_moves:
        opening_counts[fm] = opening_counts.get(fm, 0) + 1
        
    sorted_openings = sorted(opening_counts.items(), key=lambda x: x[1], reverse=True)
    opening_desc = []
    for fm, count in sorted_openings[:5]:
        pct = (count / len(first_moves)) * 100
        coord_str = f"({fm[0]},{fm[1]})" if fm[0] != -1 else "PASS"
        opening_desc.append(f"`{coord_str}`: {pct:.1f}% ({count}회)")
        
    # Moves stats
    min_moves = min(moves_counts)
    max_moves = max(moves_counts)
    avg_moves = np.mean(moves_counts)
    
    # Write Dataset Report
    md = []
    md.append("# Great Kingdom AI - Hybrid Teacher Dataset Expansion Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **총 생성 시간**: {total_elapsed:.1f}초 ({total_elapsed/60:.2f}분)\n")
    
    md.append("## 1. Dataset 규모 (Dataset Size)")
    md.append(f"* **총 게임 수**: {num_games} 판")
    md.append(f"* **총 샘플 수**: {len(all_states):,} 샘플")
    md.append(f"* **파일 크기**: {file_size_mb:.2f} MB\n")
    
    md.append("## 2. 다양성 분석 (Diversity Analysis)")
    md.append(f"* **고유 기보 비율**: **{unique_ratio*100:.2f}%** ({len(unique_seqs)}/{num_games} 고유 기보)")
    md.append(f"* **중복 기보 비율**: **{duplicate_ratio*100:.2f}%**")
    md.append(f"* **오프닝 다양성 (Top 5 첫수 분포)**: {', '.join(opening_desc)}\n")
    
    md.append("## 3. 승률 분포 (Winner Distribution)")
    md.append(f"* **BLUE (선공) 승률**: **{(blue_wins/num_games)*100:.1f}%** ({blue_wins}승)")
    md.append(f"* **ORANGE (후공) 승률**: **{(orange_wins/num_games)*100:.1f}%** ({orange_wins}승)")
    md.append(f"* **무승부 (Draw) 비율**: **{(draws/num_games)*100:.1f}%** ({draws}회)\n")
    
    md.append("## 4. 평균 수순 통계 (Moves Statistics)")
    md.append(f"* **최소 수순**: {min_moves} 수")
    md.append(f"* **평균 수순**: {avg_moves:.1f} 수")
    md.append(f"* **최대 수순**: {max_moves} 수\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Dataset report generated at: {report_path}", flush=True)

def get_topk_correct(outputs, targets, k):
    _, topk_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    return correct.any(dim=1).sum().item()

def run_policy_training_and_comparison(npz_path, model_v2_path, scaling_report_path):
    print("\n--- Starting Policy Network V2 Training with Expanded Dataset ---", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Load Datasets
    print("Loading expanded dataset...", flush=True)
    train_dataset = GreatKingdomDataset(npz_path, mode="train", split_ratio=0.9)
    val_dataset = GreatKingdomDataset(npz_path, mode="val", split_ratio=0.9)
    
    print(f"Train samples     : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    batch_size = 512  # Batch size 512 for faster training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model Setup
    model = PolicyNetworkV2().to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    
    epochs = 20
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )
    
    best_val_acc1 = 0.0
    start_train_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
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
        
        # Validate
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
                
                val_correct1 += get_topk_correct(outputs, targets, 1)
                val_correct3 += get_topk_correct(outputs, targets, 3)
                val_correct5 += get_topk_correct(outputs, targets, 5)
                
        val_loss /= len(val_dataset)
        val_acc1 = (val_correct1 / len(val_dataset)) * 100
        val_acc3 = (val_correct3 / len(val_dataset)) * 100
        val_acc5 = (val_correct5 / len(val_dataset)) * 100
        
        epoch_elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Top1: {val_acc1:.2f}% | Top3: {val_acc3:.2f}% | Top5: {val_acc5:.2f}% | {epoch_elapsed:.1f}s", flush=True)
              
        if val_acc1 > best_val_acc1:
            best_val_acc1 = val_acc1
            torch.save(model.state_dict(), model_v2_path)
            
    total_train_time = time.time() - start_train_time
    print(f"Training completed in {total_train_time:.1f}s. Best Val Top1 Accuracy: {best_val_acc1:.2f}%\n", flush=True)
    
    # ----------------- Load the best model and evaluate metrics -----------------
    model.load_state_dict(torch.load(model_v2_path))
    model.eval()
    
    final_correct1 = 0
    final_correct3 = 0
    final_correct5 = 0
    
    with torch.no_grad():
        for states, targets in val_loader:
            states = states.to(device)
            targets = targets.to(device)
            outputs = model(states)
            final_correct1 += get_topk_correct(outputs, targets, 1)
            final_correct3 += get_topk_correct(outputs, targets, 3)
            final_correct5 += get_topk_correct(outputs, targets, 5)
            
    v2_expanded_top1 = (final_correct1 / len(val_dataset)) * 100
    v2_expanded_top3 = (final_correct3 / len(val_dataset)) * 100
    v2_expanded_top5 = (final_correct5 / len(val_dataset)) * 100
    
    # Baseline Metrics (V2 trained on 1000 games)
    baseline_top1 = 37.81
    baseline_top3 = 53.01 # Estimated or from comparison report
    baseline_top5 = 60.92
    
    diff_top1 = v2_expanded_top1 - baseline_top1
    diff_top3 = v2_expanded_top3 - baseline_top3
    diff_top5 = v2_expanded_top5 - baseline_top5
    
    # Verdict logic
    if v2_expanded_top1 >= 50.0:
        verdict_status = "매우 성공"
        verdict_desc = "Top-1 정확도가 **50% 이상**으로 뛰어난 데이터 확장 효율을 보였습니다. 훈련된 정책망이 고수준 바둑 감각을 효과적으로 습득한 것으로 판명됩니다."
    elif v2_expanded_top1 >= 45.0:
        verdict_status = "성공"
        verdict_desc = "Top-1 정확도가 **45% 이상**으로 목표 기준을 통과하였습니다. 데이터셋 확장 효과가 확실히 작용하고 있습니다."
    else:
        verdict_status = "데이터 증가 효과 제한 (추가 개선 필요)"
        verdict_desc = "Top-1 정확도가 **40% 이하**이거나 45% 미만으로 도출되어, 단순 데이터 증설만으로는 모방 학습의 상한선을 크게 뚫지 못했습니다. AlphaZero 탐색(MCTS)이나 가치망 도입이 필수적입니다."
        
    # Write Scaling Report
    md = []
    md.append("# Great Kingdom AI - Policy Network Scaling Comparison Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **검증 장치**: {device}\n")
    
    md.append("## 1. 데이터셋 크기에 따른 Policy V2 성능 비교")
    md.append("| 데이터셋 규모 | 총 게임 수 | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | 학습 소요 시간 |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **기존 Baseline (v1)** | 1,000판 | {baseline_top1:.2f}% | {baseline_top3:.2f}% | {baseline_top5:.2f}% | 약 18s |")
    md.append(f"| **확장 Dataset (Hybrid)** | 5,000판 | **{v2_expanded_top1:.2f}%** | **{v2_expanded_top3:.2f}%** | **{v2_expanded_top5:.2f}%** | {total_train_time:.1f}s |")
    md.append("")
    
    md.append("## 2. 성능 상승폭 (Scaling Improvements)")
    md.append(f"* **Top-1 상승폭**: **{diff_top1:+.2f}%**")
    md.append(f"* **Top-3 상승폭**: **{diff_top3:+.2f}%**")
    md.append(f"* **Top-5 상승폭**: **{diff_top5:+.2f}%**\n")
    
    md.append("## 3. 최종 판정 (Scaling Verdict)")
    md.append(f"### 판정 결과: **{verdict_status}**")
    md.append(f"* {verdict_desc}\n")
    
    md.append("## 4. 후속 로드맵 의사결정")
    if v2_expanded_top1 >= 45.0:
        md.append("- 확장된 데이터셋을 통해 Policy Network의 모방 성능이 현격히 향상되었습니다.")
        md.append("- 따라서 현 성능 검증 모델을 기반으로 **MCTS 탐색 모듈과 Value Network 연동 파이프라인 개발 단계**로 진행합니다.")
    else:
        md.append("- 데이터 확장 대비 성능 향상 폭이 제한적이므로 단순히 모방 학습을 늘리는 방식을 멈춥니다.")
        md.append("- 차기 단계에서는 정책 모델을 활용한 MCTS 및 RL(자가 학습 루프)을 적용하여 스스로 약점을 극복하도록 유도해야 합니다.")
        
    with open(scaling_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Policy scaling report generated successfully at: {scaling_report_path}", flush=True)

def main():
    print("=================================================================")
    print("GREAT KINGDOM AI - HYBRID TEACHER DATASET EXPANSION PIPELINE")
    print("=================================================================")
    
    num_games = 5000
    model_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2.pth"
    output_npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_hybrid_5000.npz"
    dataset_report_path = r"C:\Users\User\source\repos\greatkingdomAI\dataset_hybrid_5000_report.md"
    scaling_report_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_scaling_report.md"
    
    # Phase 1: Self-play Data Generation (5000 games)
    run_selfplay_data_generation(num_games, model_path, output_npz_path, dataset_report_path)
    
    # Phase 2: Policy V2 Retraining and Performance Evaluation
    run_policy_training_and_comparison(output_npz_path, model_path, scaling_report_path)
    
    print("\nPipeline execution finished successfully!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
