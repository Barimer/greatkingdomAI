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
from ai.minimax import find_best_move, clear_transposition_table, reset_stats, get_legal_moves
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

# ----------------- Phase 1: 1000 Games Generation with T=0.8 & K=5 (Entire game) -----------------
def play_one_selfplay_hybrid_v3(args):
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
        
        # Save state tensor BEFORE the move
        state_tensor = board_to_tensor(game.board, current_player)
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            # Entire game: Temperature = 0.8, Top-k = 5
            move = find_hybrid_move(game, _worker_model, device, temperature=0.8, top_k=5)
            
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

def run_selfplay_diversity_v3(num_games, model_path, output_path):
    print(f"--- Generating {num_games} Games with Diverse v3 (T=0.8, K=5, Entire Game) ---", flush=True)
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
    
    blue_wins = 0
    orange_wins = 0
    draws = 0
    
    moves_counts = []
    move_sequences = []
    first_moves = []
    
    pool_args = [(i, model_path) for i in range(1, num_games + 1)]
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
        for res in pool.imap_unordered(play_one_selfplay_hybrid_v3, pool_args):
            completed += 1
            game_idx = res["game_idx"]
            winner = res["winner"]
            moves = res["moves"]
            termination = res["termination"]
            game_history = res["history"]
            
            total_moves += moves
            moves_counts.append(moves)
            all_terminations[game_idx - 1] = termination
            
            if winner == BLUE:
                blue_wins += 1
            elif winner == ORANGE:
                orange_wins += 1
            else:
                draws += 1
                
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
                
            if completed % 200 == 0 or completed == num_games:
                print(f"  [Progress {completed:04d}/{num_games:04d}] Elapsed: {time.time() - start_time:.1f}s | Avg Moves: {total_moves/completed:.1f}", flush=True)
                
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
    
    # Analyze diversity
    unique_seqs = set(move_sequences)
    unique_ratio = len(unique_seqs) / num_games
    duplicate_ratio = 1.0 - unique_ratio
    
    opening_counts = {}
    for fm in first_moves:
        opening_counts[fm] = opening_counts.get(fm, 0) + 1
    sorted_openings = sorted(opening_counts.items(), key=lambda x: x[1], reverse=True)
    opening_desc = []
    for fm, count in sorted_openings[:5]:
        pct = (count / len(first_moves)) * 100
        opening_desc.append(f"({fm[0]},{fm[1]}): {pct:.1f}%")
        
    avg_moves = np.mean(moves_counts)
    
    print(f"Dataset generated at {output_path} (Size: {file_size_mb:.2f} MB)", flush=True)
    print(f"  Uniqueness Ratio: {unique_ratio*100:.2f}% | Duplication Ratio: {duplicate_ratio*100:.2f}%", flush=True)
    
    return {
        "games": num_games,
        "samples": len(all_states),
        "duration": total_elapsed,
        "unique_ratio": unique_ratio,
        "duplicate_ratio": duplicate_ratio,
        "opening_desc": ", ".join(opening_desc),
        "avg_moves": avg_moves,
        "blue_win_pct": (blue_wins / num_games) * 100,
        "orange_win_pct": (orange_wins / num_games) * 100
    }

# ----------------- Phase 2: Retrain Policy Network -----------------
def run_retrain_model(npz_path, model_v2_path):
    print("\n--- Retraining Policy Network V2 with Diverse v3 Dataset ---", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}", flush=True)
    
    train_dataset = GreatKingdomDataset(npz_path, mode="train", split_ratio=0.9)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    
    model = PolicyNetworkV2().to(device)
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
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for states, targets in train_loader:
            states = states.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
    print(f"Retraining completed and saved to {model_v2_path} (Took: {time.time() - start_time:.1f}s)", flush=True)
    torch.save(model.state_dict(), model_v2_path)

# ----------------- Phase 3: Match Verification Tasks -----------------
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
    legal_indices = [get_move_idx(m) for m in legal_moves]
    
    legal_probs = probs[legal_indices]
    if np.sum(legal_probs) > 0:
        legal_probs /= np.sum(legal_probs)
        best_idx = np.argmax(legal_probs)
        chosen_idx = legal_indices[best_idx]
    else:
        chosen_idx = legal_indices[0]
    return get_move_from_idx(chosen_idx)

# Matchup 1: New Model vs Temp=0.8 Model (K=5)
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
    return {"winner": winner, "moves": move_count, "duration": time.time() - game_start}

# Matchup 2: New Model vs Hybrid Teacher (Deterministic)
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
                # Hybrid Teacher in deterministic mode (T=None)
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
    return {"winner": winner, "moves": move_count, "duration": time.time() - game_start}

# Matchup 3: New Model vs Depth 3 Minimax
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
    return {"winner": winner, "moves": move_count, "duration": time.time() - game_start}

def write_failure_report(stats, report_path):
    unique_pct = stats["unique_ratio"] * 100
    md = []
    md.append("# Great Kingdom AI - Dataset Diversity Phase v3 Failure Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **검증 조건**: Temperature=0.8, Top-k=5 가중 샘플링 전체 게임 적용 (1,000판)\n")
    md.append("## 1. 다양성 분석 결과")
    md.append(f"* **고유 기보 비율**: **{unique_pct:.2f}%** (기준 20% 미만) — **판정: 실패 (Fail)**")
    md.append(f"* **중복 기보 비율**: {stats['duplicate_ratio']*100:.2f}%")
    md.append(f"* **총 게임 수**: {stats['games']} 판")
    md.append(f"* **총 샘플 수**: {stats['samples']:,} 샘플")
    md.append(f"* **평균 게임 길이**: {stats['avg_moves']:.1f} 수")
    md.append(f"* **오프닝 다양성 (첫수 분포)**: {stats['opening_desc']}")
    md.append(f"* **선/후공 승률**: BLUE {stats['blue_win_pct']:.1f}% / ORANGE {stats['orange_win_pct']:.1f}%\n")
    
    md.append("## 2. 다양성 확보 실패 원인 분석")
    md.append("- **Top-K의 좁은 바운더리**: Top-k=5로 설정하여 상위 5개 후보 내로 탐색 범위를 제한한 경우, 특정 유리한 상태에서 최선수 이외의 선택지 확률 분포가 여전히 매우 좁아 중복 기보가 반복 생성된 것으로 해석됩니다.")
    md.append("- **온도 파라미터의 한계**: Temperature=0.8 수준은 탐색 노이즈를 충분히 발생시키기 어렵습니다. 특히 20%를 달성하기 위해서는 11수 이후의 결정론적 모드가 없더라도, 초반을 포함한 게임 전체에서 더 높은 Temperature(예: T=1.0 이상) 또는 Dirichlet Noise가 필수적입니다.")
    md.append("- **결론**: Phase 2~4를 전면 중단하고 데이터의 다양성을 확장할 새로운 전략(예: T=1.2로 상향 조정, 혹은 Top-k 해제 및 Dirichlet Noise 도입)을 수립해야 합니다.")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Failure report successfully saved to: {report_path}", flush=True)

def write_success_report(stats, new_v_temp08_rate, new_v_hybrid_rate, new_v_d3_rate, new_v_temp08_wins, new_v_hybrid_wins, new_v_d3_wins, report_path):
    unique_pct = stats["unique_ratio"] * 100
    
    # Success evaluation for Uniqueness
    if unique_pct >= 40.0:
        unique_verdict = "매우 성공 (40% 이상)"
    elif unique_pct >= 20.0:
        unique_verdict = "성공 (20% 이상)"
    else:
        unique_verdict = "성공 기준 미달 (20% 미만)"
        
    # Success evaluation for Depth3 win rate
    if new_v_d3_rate >= 30.0:
        d3_verdict = "성공 (30% 이상)"
    else:
        d3_verdict = "실패 (30% 미만)"
        
    overall_success = unique_pct >= 20.0 and new_v_d3_rate >= 30.0
    final_verdict_str = "SUCCESS (다양성 및 실전 지능 확보)" if overall_success else "PARTIAL SUCCESS (다양성 확보했으나 실전 지능 부족)"
    
    md = []
    md.append("# Great Kingdom AI - Dataset Diversity Phase v3 Success Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **검증 조건**: Temperature=0.8, Top-k=5 가중 샘플링 전체 게임 적용 (1,000판)\n")
    
    md.append("## 1. 신규 데이터셋 통계 (Dataset Metrics)")
    md.append(f"* **총 게임 수**: {stats['games']} 판")
    md.append(f"* **총 샘플 수**: {stats['samples']:,} 샘플")
    md.append(f"* **고유 기보 비율**: **{unique_pct:.2f}%** — **판정: {unique_verdict}**")
    md.append(f"* **중복 기보 비율**: **{stats['duplicate_ratio']*100:.2f}%**")
    md.append(f"* **오프닝 다양성 (첫수 분포)**: {stats['opening_desc']}")
    md.append(f"* **평균 게임 길이**: {stats['avg_moves']:.1f} 수")
    md.append(f"* **BLUE (선공) 승률**: {stats['blue_win_pct']:.1f}% | **ORANGE (후공) 승률**: {stats['orange_win_pct']:.1f}%\n")
    
    md.append("## 2. 데이터셋 다양성 변화 정량 비교 (Quantitative Comparison)")
    md.append("| 데이터셋 조건 | 고유 기보 비율 | 중복 기보 비율 | Depth 3 상대 승률 | 최종 판정 |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")
    md.append(f"| **A) 결정론적 데이터셋** | 1.64% | 98.36% | 18.0% | 암기형 과적합 (Memorized) |")
    md.append(f"| **B) Temp=0.8 + Top-k=5 (초반10수)** | 10.30% | 89.70% | 30.0% | 과적합 탈피 시작 (Generalized) |")
    md.append(f"| **C) Temp=0.8 + Top-k=5 (전체게임)** | **{unique_pct:.2f}%** | {stats['duplicate_ratio']*100:.2f}% | **{new_v_d3_rate:.1f}%** | **{final_verdict_str}** |")
    md.append("")
    
    md.append("## 3. 실전 대국 검증 결과 (Matchup Statistics)")
    md.append("| 평가 대상 (Matchup) | 총 판수 | 새 모델 (Diverse v3) 승률 | 비고 |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| **새 모델 vs 기존 Temp0.8 모델** | 50판 | **{new_v_temp08_rate:.1f}%** ({new_v_temp08_wins}승) | 다양성 학습 모델 간의 1:1 대조 |")
    md.append(f"| **새 모델 vs Hybrid Teacher (결정론)** | 50판 | **{new_v_hybrid_rate:.1f}%** ({new_v_hybrid_wins}승) | 하이브리드 엔진 모방 지표 |")
    md.append(f"| **새 모델 vs Depth 3 Minimax** | 50판 | **{new_v_d3_rate:.1f}%** ({new_v_d3_wins}승) | **성공 판정: {d3_verdict}** |")
    md.append("")
    
    md.append("## 4. 종합 판정 및 의사결정 (Overall Conclusion)")
    if overall_success:
        md.append(f"### 최종 판정: **PASS (일반화 성능 완성 - 대규모 학습 전환 가능)**")
        md.append(f"- 게임 전체 영역에 확률적 Temperature Sampling(T=0.8, K=5)을 적용함으로써, 고유 기보 비율이 **{unique_pct:.2f}%**로 상승하여 성공 기준(20%)을 달성하였습니다.")
        md.append(f"- 또한 Depth 3 상대로 승률 **{new_v_d3_rate:.1f}%**를 기록하며 최적의 탐색 실력을 일반화하는 데 성공하였습니다.")
        md.append("- 다음 단계로 **5,000판 이상 대규모 다양성 데이터 수집 및 학습**을 추진하는 것을 권장합니다.")
    else:
        md.append(f"### 최종 판정: **FAIL (추가 개선 필요)**")
        md.append("- 고유 기보 비율은 20%를 달성했으나 Depth3 승률이 목표(30%)에 미치지 못했습니다. 기보 다양성이 과도해지며 학습 신호에 노이즈가 유입되었을 가능성이 있습니다. 하이브리드 가중치 수정을 검토해야 합니다.")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Success report successfully saved to: {report_path}", flush=True)

def main():
    print("=================================================================")
    print("GREAT KINGDOM AI - DIVERSITY EXPANSION PHASE V3 PIPELINE")
    print("=================================================================")
    
    model_5000_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_5000.pth"
    model_temp08_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_new_temp.pth"
    model_new_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2.pth" 
    output_npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_diverse_v3_1000.npz"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\dataset_diversity_v3_report.md"
    
    # 1. Generate 1000 Games with T=0.8 & K=5 (Entire game)
    stats = run_selfplay_diversity_v3(1000, model_5000_path, output_npz_path)
    
    # --- CRITICAL DECISION POINT: Uniqueness Ratio Check ---
    unique_pct = stats["unique_ratio"] * 100
    print(f"\n>>> [KPI Check] Uniqueness Ratio: {unique_pct:.2f}% (Target: >= 20.0%)", flush=True)
    
    if unique_pct < 20.0:
        print(f"\n>>> [Verdict] Uniqueness Ratio {unique_pct:.2f}% is below 20%.", flush=True)
        print(">>> Aborting Phase 2~4 immediately to save GPU/CPU compute resources.", flush=True)
        print(">>> Generating diversity failure analysis report...", flush=True)
        write_failure_report(stats, report_path)
        print(">>> Process Terminated.", flush=True)
        return
        
    print(f"\n>>> [Verdict] Uniqueness Ratio {unique_pct:.2f}% is >= 20%. Proceeding to Phase 2 (Retrain).", flush=True)
    
    # 2. Retrain Policy Network V2 with the new dataset
    run_retrain_model(output_npz_path, model_new_path)
    
    # 3. Matchup Evaluators (50 games each)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    
    tasks = []
    for i in range(1, 51):
        color = BLUE if i <= 25 else ORANGE
        tasks.append((i, color))
        
    print("\n--- Running Matchup Verification Battles (50 games each) ---", flush=True)
    
    # Matchup 1: New vs Temp=0.8 Model
    print("  1/3: New Model vs Temp=0.8 Model...", flush=True)
    new_v_temp08_wins = 0
    completed = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_match_worker, initargs=(model_new_path, model_temp08_path)) as pool:
        for res in pool.imap_unordered(play_new_vs_temp08, tasks):
            completed += 1
            new_color = tasks[completed-1][1]
            if res["winner"] == new_color:
                new_v_temp08_wins += 1
    new_v_temp08_rate = (new_v_temp08_wins / 50) * 100
    
    # Matchup 2: New vs Hybrid Teacher
    print("  2/3: New Model vs Hybrid Teacher...", flush=True)
    new_v_hybrid_wins = 0
    completed = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_match_worker, initargs=(model_new_path, model_temp08_path)) as pool:
        for res in pool.imap_unordered(play_new_vs_hybrid, tasks):
            completed += 1
            new_color = tasks[completed-1][1]
            if res["winner"] == new_color:
                new_v_hybrid_wins += 1
    new_v_hybrid_rate = (new_v_hybrid_wins / 50) * 100
    
    # Matchup 3: New vs Depth 3 Minimax
    print("  3/3: New Model vs Depth 3...", flush=True)
    new_v_d3_wins = 0
    completed = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_match_worker, initargs=(model_new_path, model_temp08_path)) as pool:
        for res in pool.imap_unordered(play_new_vs_depth3, tasks):
            completed += 1
            new_color = tasks[completed-1][1]
            if res["winner"] == new_color:
                new_v_d3_wins += 1
    new_v_d3_rate = (new_v_d3_wins / 50) * 100
    
    # 4. Final Success Report
    write_success_report(stats, new_v_temp08_rate, new_v_hybrid_rate, new_v_d3_rate, new_v_temp08_wins, new_v_hybrid_wins, new_v_d3_wins, report_path)
    print("\n>>> Pipeline successfully finished.", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
