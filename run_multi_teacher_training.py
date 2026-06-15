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

def get_topk_correct(outputs, targets, k):
    _, topk_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    return correct.any(dim=1).sum().item()

# ----------------- Phase 1: Retrain Model -----------------
def run_retrain_model_multi_teacher(npz_path, model_path):
    print("\n--- Phase 1: Retraining Policy Network V2 with Multi-Teacher Dataset ---", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training device: {device}", flush=True)
    
    train_dataset = GreatKingdomDataset(npz_path, mode="train", split_ratio=0.9)
    val_dataset = GreatKingdomDataset(npz_path, mode="val", split_ratio=0.9)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    
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
    
    history = []
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Training Phase
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
        
        # Validation Phase
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
        
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc1: {val_acc1:.2f}% | Val Acc3: {val_acc3:.2f}% | Val Acc5: {val_acc5:.2f}%", flush=True)
        
    total_elapsed = time.time() - start_time
    print(f"Retraining completed (Saved to {model_path}) | Took: {total_elapsed:.1f}s", flush=True)
    torch.save(model.state_dict(), model_path)
    
    return history

# ----------------- Phase 2: Match Verification Setup -----------------
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

def main():
    print("=================================================================")
    print("GREAT KINGDOM AI - MULTI-TEACHER TRAINING & EVALUATION PIPELINE")
    print("=================================================================")
    
    model_temp08_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_new_temp.pth"
    model_new_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_multi_teacher.pth"
    npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_multi_teacher_1000.npz"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\multi_teacher_training_report.md"
    
    # 1. Retrain Policy Network V2 with the Multi-Teacher dataset
    history = run_retrain_model_multi_teacher(npz_path, model_new_path)
    
    # 2. Matchup Evaluators (50 games each)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    
    tasks = []
    for i in range(1, 51):
        color = BLUE if i <= 25 else ORANGE
        tasks.append((i, color))
        
    print("\n--- Phase 2: Running Matchup Verification Battles (50 games each) ---", flush=True)
    
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
    final_verdict_str = "SUCCESS (일반화 성능 향상 입증)" if overall_success else "FAIL (다양성은 올랐으나 실력 개선 미비)"
    
    # Write Final report
    md = []
    md.append("# Great Kingdom AI - Multi-Teacher Training & Evaluation Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **학습 조건**: Multi-Teacher 혼합 데이터셋 (42,939 샘플, 36.00% Diversity)\n")
    
    md.append("## 1. 정책 네트워크(Policy Network) 학습 결과 기록")
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
    
    md.append("## 2. 실전 대국 검증 결과 (Matchup Statistics)")
    md.append("| 평가 대상 (Matchup) | 총 판수 | 새 모델 (Multi-Teacher) 승률 | 비고 |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| **새 모델 vs 기존 Temp0.8 모델** | 50판 | **{new_v_temp08_rate:.1f}%** ({new_v_temp08_wins}승) | 다양성 학습 모델 간의 대조 |")
    md.append(f"| **새 모델 vs Hybrid Teacher (결정론)** | 50판 | **{new_v_hybrid_rate:.1f}%** ({new_v_hybrid_wins}승) | 하이브리드 엔진 모방 지표 |")
    md.append(f"| **새 모델 vs Depth 3 Minimax** | 50판 | **{new_v_d3_rate:.1f}%** ({new_v_d3_wins}승) | **성공 판정: {d3_verdict}** |")
    md.append("")
    
    md.append("## 3. 다양성 극대화 모델 vs 기존 모델 비교 분석")
    md.append("| 지표 (Metrics) | 기존 최고 모델 (Temp0.8, K=5 초반) | 신규 Multi-Teacher 모델 | 개선 폭 |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **데이터셋 고유 기보 비율** | 10.30% | **36.00%** | **+25.70%p** |")
    md.append(f"| **Depth 3 상대 승률** | 30.0% | **{new_v_d3_rate:.1f}%** | **{new_v_d3_rate - 30.0:+.1f}%p** |")
    md.append("")
    
    md.append("## 4. 최종 질문에 대한 해답: 36% 다양성이 실력 향상으로 이어지는가?")
    md.append(f"### 최종 판정: **{final_verdict_str}**")
    if overall_success:
        md.append(f"- **실험 결과**: 36% 다양성을 가진 데이터셋으로의 학습이 **실제 실력 향상으로 강하게 직결됨**이 실험적으로 검증되었습니다.")
        md.append(f"- 기존 최고 모델(Depth3 상대 승률 30.0%) 대비 **승률이 {new_v_d3_rate:.1f}%**로 반등하며 성공 기준(35% 이상)을 만족시켰습니다.")
        md.append("- 이는 단순 모방 학습이 데이터셋의 고유 기보 비중이 높아짐에 따라 '암기(Memorization)'에서 벗어나 바둑/체스형 상태의 전술적 패턴을 '일반화(Generalization)'하기 시작했음을 가리킵니다.")
    else:
        md.append("- **실험 결과**: 다양성은 36%로 크게 확대되었으나, Depth3 상대 승률은 기존(30%) 대비 크게 개선되지 못했습니다.")
        md.append("- 이는 다중 교사 혼합 시 각 교사들이 제공하는 가이드에 모순이 발생하여(예: 탐색 깊이 차이에 의한 상반된 수) 정책 학습 신호에 노이즈가 과해졌기 때문일 수 있습니다.")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Training report successfully saved to: {report_path}", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
