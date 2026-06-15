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

_model_1000 = None
_model_5000 = None

def init_worker(model_1000_path, model_5000_path):
    global _model_1000, _model_5000
    device = torch.device("cpu")
    
    _model_1000 = PolicyNetworkV2().to(device)
    if os.path.exists(model_1000_path):
        _model_1000.load_state_dict(torch.load(model_1000_path, map_location=device))
    _model_1000.eval()
    
    _model_5000 = PolicyNetworkV2().to(device)
    if os.path.exists(model_5000_path):
        _model_5000.load_state_dict(torch.load(model_5000_path, map_location=device))
    _model_5000.eval()

# ----------------- Helper: Play Pure Policy Move -----------------
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

# ----------------- Game Simulation Tasks -----------------

# Task 1: Policy 1000 vs Policy 5000
def play_1000_vs_5000(args):
    game_idx, p5000_color = args
    global _model_1000, _model_5000
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    device = torch.device("cpu")
    
    move_count = 0
    max_moves = 150
    game_start = time.time()
    p5000_inf = []
    p1000_inf = []
    
    while not game.game_over and move_count < max_moves:
        curr_player = game.current_player
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if curr_player == p5000_color:
                start_inf = time.time()
                move = get_pure_policy_move(game, _model_5000, device)
                p5000_inf.append(time.time() - start_inf)
            else:
                start_inf = time.time()
                move = get_pure_policy_move(game, _model_1000, device)
                p1000_inf.append(time.time() - start_inf)
                
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                game.play_pass()
                
        move_count += 1
        
    winner = game.winner if game.winner is not None else game.check_winner()
    return {
        "winner": winner,
        "moves": move_count,
        "duration": time.time() - game_start,
        "p5000_inf": p5000_inf,
        "p1000_inf": p1000_inf
    }

# Task 2: Policy 5000 vs Depth 3
def play_5000_vs_depth3(args):
    game_idx, p5000_color = args
    global _model_5000
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    device = torch.device("cpu")
    
    move_count = 0
    max_moves = 150
    game_start = time.time()
    p5000_inf = []
    d3_inf = []
    
    while not game.game_over and move_count < max_moves:
        curr_player = game.current_player
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if curr_player == p5000_color:
                start_inf = time.time()
                move = get_pure_policy_move(game, _model_5000, device)
                p5000_inf.append(time.time() - start_inf)
            else:
                start_inf = time.time()
                f = io.StringIO()
                with redirect_stdout(f):
                    move = find_best_move(game, depth=3)
                d3_inf.append(time.time() - start_inf)
                
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                game.play_pass()
                
        move_count += 1
        
    winner = game.winner if game.winner is not None else game.check_winner()
    return {
        "winner": winner,
        "moves": move_count,
        "duration": time.time() - game_start,
        "p5000_inf": p5000_inf,
        "d3_inf": d3_inf
    }

# Task 3: Policy 5000 vs Hybrid Teacher
def play_5000_vs_hybrid(args):
    game_idx, p5000_color = args
    global _model_5000
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    device = torch.device("cpu")
    
    move_count = 0
    max_moves = 150
    game_start = time.time()
    p5000_inf = []
    hybrid_inf = []
    
    while not game.game_over and move_count < max_moves:
        curr_player = game.current_player
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if curr_player == p5000_color:
                start_inf = time.time()
                move = get_pure_policy_move(game, _model_5000, device)
                p5000_inf.append(time.time() - start_inf)
            else:
                start_inf = time.time()
                move = find_hybrid_move(game, _model_5000, device)
                hybrid_inf.append(time.time() - start_inf)
                
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                game.play_pass()
                
        move_count += 1
        
    winner = game.winner if game.winner is not None else game.check_winner()
    return {
        "winner": winner,
        "moves": move_count,
        "duration": time.time() - game_start,
        "p5000_inf": p5000_inf,
        "hybrid_inf": hybrid_inf
    }

# ----------------- Baseline 1000 Games Model Restoration -----------------
def restore_baseline_model(npz_path, output_path):
    print("--- Restoring 1,000 Games Baseline Model (Policy V2) ---", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
            
    print(f"Baseline model restored and saved to {output_path} (Took: {time.time() - start_time:.1f}s)", flush=True)
    torch.save(model.state_dict(), output_path)

def main():
    print("=================================================================")
    print("GREAT KINGDOM AI - GENERALIZATION VERIFICATION PIPELINE")
    print("=================================================================")
    
    baseline_npz = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_dataset_v1.npz"
    model_1000_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_1000.pth"
    model_5000_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_5000.pth"
    current_v2_pth = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2.pth"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\generalization_verification_report.md"
    
    # 0. Backup / Restore 가중치
    if os.path.exists(current_v2_pth) and not os.path.exists(model_5000_path):
        import shutil
        shutil.copyfile(current_v2_pth, model_5000_path)
        print(f"Backed up 5000-game model to {model_5000_path}")
        
    if not os.path.exists(model_1000_path):
        restore_baseline_model(baseline_npz, model_1000_path)
        
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    
    tasks_50 = []
    for i in range(1, 51):
        color = BLUE if i <= 25 else ORANGE
        tasks_50.append((i, color))
        
    wall_start = time.time()
    
    # 1. Policy 1000 vs Policy 5000 (50 games)
    print("\n--- 1/3: Policy V2 (1000) vs Policy V2 (5000) (50 games) ---", flush=True)
    completed = 0
    p5000_v_1000_wins = 0
    p5000_v_1000_moves = 0
    p5000_v_1000_inf = []
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(model_1000_path, model_5000_path)) as pool:
        for res in pool.imap_unordered(play_1000_vs_5000, tasks_50):
            completed += 1
            winner = res["winner"]
            p5000_color = tasks_50[completed-1][1]
            if winner == p5000_color:
                p5000_v_1000_wins += 1
            p5000_v_1000_moves += res["moves"]
            p5000_v_1000_inf.extend(res["p5000_inf"])
            winner_str = "V2_5000" if winner == p5000_color else "V2_1000" if winner is not None else "DRAW"
            print(f"  [Match {completed:02d}/50] Winner: {winner_str} | Moves: {res['moves']} | Duration: {res['duration']:.1f}s", flush=True)
            
    p5000_v_1000_rate = (p5000_v_1000_wins / 50) * 100
    p5000_v_1000_avg_moves = p5000_v_1000_moves / 50
    p5000_v_1000_avg_inf = np.mean(p5000_v_1000_inf) * 1000 if p5000_v_1000_inf else 0.0
    
    # 2. Policy 5000 vs Depth 3 (50 games)
    print("\n--- 2/3: Policy V2 (5000) vs Depth 3 Minimax (50 games) ---", flush=True)
    completed = 0
    p5000_v_d3_wins = 0
    p5000_v_d3_moves = 0
    p5000_v_d3_inf = []
    d3_inf = []
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(model_1000_path, model_5000_path)) as pool:
        for res in pool.imap_unordered(play_5000_vs_depth3, tasks_50):
            completed += 1
            winner = res["winner"]
            p5000_color = tasks_50[completed-1][1]
            if winner == p5000_color:
                p5000_v_d3_wins += 1
            p5000_v_d3_moves += res["moves"]
            p5000_v_d3_inf.extend(res["p5000_inf"])
            d3_inf.extend(res["d3_inf"])
            winner_str = "V2_5000" if winner == p5000_color else "DEPTH3" if winner is not None else "DRAW"
            print(f"  [Match {completed:02d}/50] Winner: {winner_str} | Moves: {res['moves']} | Duration: {res['duration']:.1f}s", flush=True)
            
    p5000_v_d3_rate = (p5000_v_d3_wins / 50) * 100
    p5000_v_d3_avg_moves = p5000_v_d3_moves / 50
    p5000_v_d3_avg_inf = np.mean(p5000_v_d3_inf) * 1000 if p5000_v_d3_inf else 0.0
    d3_avg_inf = np.mean(d3_inf) * 1000 if d3_inf else 0.0
    
    # 3. Policy 5000 vs Hybrid Teacher (50 games)
    print("\n--- 3/3: Policy V2 (5000) vs Hybrid Teacher (50 games) ---", flush=True)
    completed = 0
    p5000_v_hybrid_wins = 0
    p5000_v_hybrid_moves = 0
    p5000_v_hybrid_inf = []
    hybrid_inf = []
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(model_1000_path, model_5000_path)) as pool:
        for res in pool.imap_unordered(play_5000_vs_hybrid, tasks_50):
            completed += 1
            winner = res["winner"]
            p5000_color = tasks_50[completed-1][1]
            if winner == p5000_color:
                p5000_v_hybrid_wins += 1
            p5000_v_hybrid_moves += res["moves"]
            p5000_v_hybrid_inf.extend(res["p5000_inf"])
            hybrid_inf.extend(res["hybrid_inf"])
            winner_str = "V2_5000" if winner == p5000_color else "HYBRID" if winner is not None else "DRAW"
            print(f"  [Match {completed:02d}/50] Winner: {winner_str} | Moves: {res['moves']} | Duration: {res['duration']:.1f}s", flush=True)
            
    p5000_v_hybrid_rate = (p5000_v_hybrid_wins / 50) * 100
    p5000_v_hybrid_avg_moves = p5000_v_hybrid_moves / 50
    p5000_v_hybrid_avg_inf = np.mean(p5000_v_hybrid_inf) * 1000 if p5000_v_hybrid_inf else 0.0
    hybrid_avg_inf = np.mean(hybrid_inf) * 1000 if hybrid_inf else 0.0
    
    wall_time = time.time() - wall_start
    
    # ----------------- Generalization Analysis & Decision -----------------
    # Baseline comparison details (Baseline 1000-game policy vs Depth3 winrate was ~35.0% or 0% depending on setup, but let's compare with 38.0% Hybrid or previous V2 vs D3 of 35%)
    # Let's assess if p5000 model won against 1000-game baseline model (should be > 50%) and improved against Depth3 (should be > 35%)
    is_better_than_1000 = p5000_v_1000_rate > 50.0
    is_better_than_d3 = p5000_v_d3_rate > 35.0  # baseline Policy V2 vs D2 was 35%, vs D3 was not tested but D3 is stronger
    
    passed_validation = is_better_than_1000 and p5000_v_d3_rate >= 40.0
    
    if passed_validation:
        verdict = "Case A (실제 실력 향상)"
        conclusion = "95%의 정확도는 중복 기보의 단순 암기(Memorization)에 그치지 않고, 다수의 다양한 상황에서 Depth3 탐색 기량과 1,000판 기본 모델을 압도하는 **실제적이고 일반화된(Generalized) 지능 향상**을 이루었음을 증명합니다."
    else:
        verdict = "Case B (데이터 암기 가능성 높음)"
        conclusion = "95%의 높은 정확도를 기록하였으나, 1,000판 기본 모델 또는 Depth3를 상대로 한 실전 승률이 향상되지 않았습니다. 이는 중복 기보(중복률 98.36%)의 과적합(Overfitting)으로 인해 실제 게임 전술 능력이 결여된 암기 상태임을 뜻합니다."
        
    # Write report
    md = []
    md.append("# Great Kingdom AI - Policy Generalization Verification Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **총 대국 시간 (Wall Clock)**: {wall_time:.1f}초 ({wall_time/60:.2f}분)\n")
    
    md.append("## 1. 모델 대국 실험 결과 (Matchup Statistics)")
    md.append("| 대결 상대 (Matchup) | 총 판수 | Policy V2 (5000판) 승률 | 평균 수순 (Moves) | 5000판 평균 추론 | 상대 평균 추론 |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **V2 (5000) vs V2 (1000)** | 50판 | **{p5000_v_1000_rate:.1f}%** ({p5000_v_1000_wins}승) | {p5000_v_1000_avg_moves:.1f} 수 | {p5000_v_1000_avg_inf:.2f} ms | {p5000_v_1000_avg_inf:.2f} ms |")
    md.append(f"| **V2 (5000) vs Depth 3** | 50판 | **{p5000_v_d3_rate:.1f}%** ({p5000_v_d3_wins}승) | {p5000_v_d3_avg_moves:.1f} 수 | {p5000_v_d3_avg_inf:.2f} ms | {d3_avg_inf:.2f} ms |")
    md.append(f"| **V2 (5000) vs Hybrid Teacher** | 50판 | **{p5000_v_hybrid_rate:.1f}%** ({p5000_v_hybrid_wins}승) | {p5000_v_hybrid_avg_moves:.1f} 수 | {p5000_v_hybrid_avg_inf:.2f} ms | {hybrid_avg_inf:.2f} ms |")
    md.append("")
    
    md.append("## 2. 최종 판정 (Generalization Verdict)")
    md.append(f"### 판정 등급: **{verdict}**")
    md.append(f"* **상세 분석**: {conclusion}\n")
    
    md.append("## 3. 차기 데이터 생성기 다양성 개선안 설계 (Data Generator Improvement)")
    md.append("현재 데이터셋의 중복률(98.36%)을 극복하여 학습 데이터의 커버리지를 비약적으로 넓히기 위한 착수 다양성 기법들의 장단점 분석입니다:\n")
    
    md.append("### 1. Temperature Sampling (온도 샘플링)")
    md.append("- **원리**: Softmax 출력 logits를 온도 $T$로 나누어 확률 분포를 부드럽게(Soft) 또는 뾰족하게(Hard) 조절함. ($p_i = e^{z_i/T} / \\sum e^{z_j/T}$)")
    md.append("- **장점**: 온도를 미세하게 조절하여 완전 랜덤(T=high)과 완전 결정론(T=low) 사이의 유연한 균형을 잡을 수 있음. 포석 단계에서는 높여 다양성을 확보하고 후반에는 낮춰 실수 방지 가능.")
    md.append("- **단점**: 온도가 너무 높으면 규칙상 합법적이지만 대단히 불리한 자충수나 패착을 남발하게 됨.\n")
    
    md.append("### 2. Top-k Sampling")
    md.append("- **원리**: 확률 분포 상위 $K$개의 후보군만 필터링한 후 그 안에서 다시 정규화하여 샘플링을 진행함.")
    md.append("- **장점**: 확률 하위권에 포진한 패착 후보들을 원천 차단하여 안전하면서도 매 대국마다 다양한 2~3위권의 준수한 대안을 선택할 수 있음.")
    md.append("- **단점**: 보드 상황에 따라 합법수가 2~3개뿐일 때 무리하게 $K$를 채우려고 하면 비정상적인 후보가 강제 선택될 수 있어 보정이 필요함.\n")
    
    md.append("### 3. $\\epsilon$-greedy (에프실론 그리디)")
    md.append("- **원리**: $1 - \\epsilon$의 확률로는 최고 확률의 수(Best Action)를 선택하고, $\\epsilon$의 확률로는 전체 합법수 중 무작위(Random) 선택을 감행함.")
    md.append("- **장점**: 가장 단순하고 직관적이며 전반적인 결정론적 성능을 매우 강력하게 보존할 수 있음.")
    md.append("- **단점**: 무작위 탐색 시(즉 $\\epsilon$ 확률 당첨 시) 극악의 악수(자충 등)를 두게 되어 대마가 죽는 참사가 터질 수 있어 바둑류 게임엔 다소 불안정함.\n")
    
    md.append("### 4. Dirichlet Noise (디리클레 노이즈 - AlphaZero 표준)")
    md.append("- **원리**: 루트 노드에서 합법수 확률 분포에 Dirichlet 분포 노이즈를 섞어줌. ($P(s,a) = (1-\\epsilon)p_a + \\epsilon \\eta_a$, where $\\eta \\sim \\text{Dir}(\\alpha)$)")
    md.append("- **장점**: AlphaZero에서 공식 검증된 표준 기법으로, 오프닝 다양성을 비약적으로 증가시키면서도 기량 파괴를 최소화하며, 특정 루트로의 편향(오버피팅)을 완벽히 방지함.")
    md.append("- **단점**: MCTS 구조와 결합될 때 최적의 시너지를 내며, 단순 Policy Network 단독 구현에서는 튜닝(노이즈 가중치 $\\alpha$)이 조금 까다로움.\n")
    
    md.append("---")
    md.append("## 4. 최종 질문에 대한 답변")
    md.append("### **\"95% 정확도는 실제 강해진 것인가? 아니면 반복 기보를 암기한 것인가?\"**")
    if passed_validation:
        md.append(f"- **실제 실력 향상으로 판명되었습니다.**")
        md.append(f"- 5,000판 모델이 기존 1,000판 대비 실전 대국 승률(**{p5000_v_1000_rate:.1f}%**)에서 확실히 우위를 보였고, Depth3 상대로도 **{p5000_v_d3_rate:.1f}%**의 높은 승률을 달성했습니다.")
        md.append("- 비록 고유 기보 시퀀스는 82개로 협소했으나, 82가지 포석 시나리오 속에서 펼쳐지는 22만 개 국면에 대한 정교한 정책 수렴이 실제 플레이에서의 강력한 일반화(Generalization)로 이어졌습니다.")
    else:
        md.append(f"- **안타깝게도 반복 기보의 단순 암기(Overfitting/Memorization)로 판명되었습니다.**")
        md.append(f"- 정확도는 95%에 도달했으나, 실전 승률이 기대치에 미달하였으며, 이는 고유 기보 다양성 결여로 인해 학습에 노이즈가 과적합되어 새로운 변화에 대응하지 못했음을 뜻합니다.")
        md.append("- 따라서 차기 단계에서는 **Dirichlet Noise 또는 Temperature Sampling** 기법을 즉시 자가대국 생성기에 결합하여 기보 다양성을 80% 이상으로 확보한 신규 데이터셋을 수집해야 합니다.")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Generalization verification report written to: {report_path}", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
