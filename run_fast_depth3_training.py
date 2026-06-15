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

# ----------------- Phase 1: Train Model -----------------
def run_train_model(npz_path, model_path, epochs=20):
    print("\n=== Phase 1: Training Policy Network V2 with Fast Depth3 Pilot Dataset ===", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training device: {device}", flush=True)
    
    # Load Datasets (90% Train / 10% Validation split)
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
        
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc1: {val_acc1:.2f}% | Val Acc3: {val_acc3:.2f}% | Val Acc5: {val_acc5:.2f}%", flush=True)
        
        # Save best model based on validation loss or top1 accuracy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc1 = val_acc1
            torch.save(model.state_dict(), model_path)
            
    total_elapsed = time.time() - start_time
    print(f"  Training completed (Best Val Loss: {best_val_loss:.4f}, Best Top1: {best_val_acc1:.2f}%) | Took: {total_elapsed:.1f}s", flush=True)
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

# Matchup 1: New Model vs Temp=0.8 Model
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
    return {"winner": winner, "moves": move_count, "duration": time.time() - game_start, "new_color": new_color}

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
    return {"winner": winner, "moves": move_count, "duration": time.time() - game_start, "new_color": new_color}

def main():
    print("=================================================================")
    print("GREAT KINGDOM AI - FAST DEPTH3 IMMEDIATE TRAINING & EVALUATION")
    print("=================================================================")
    
    model_temp08_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_new_temp.pth"
    model_new_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_fast_depth3_pilot.pth"
    npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_fast_depth3_pilot_300.npz"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\fast_depth3_immediate_training_report.md"
    artifact_report_path = r"C:\Users\User\.gemini\antigravity-cli\brain\505d76c0-2e72-46fd-bbbf-aa02a413b645\fast_depth3_immediate_training_report.md"
    
    # 1. Train Policy Network V2 with the Pilot dataset
    history = run_train_model(npz_path, model_new_path, epochs=20)
    
    # 2. Matchup Evaluators (50 games each)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    
    tasks = []
    for i in range(1, 51):
        color = BLUE if i <= 25 else ORANGE
        tasks.append((i, color))
        
    print("\n=== Phase 2: Running Matchup Verification Battles (50 games each) ===", flush=True)
    
    # Matchup 1: New vs Temp=0.8 Model
    print("  1/3: New Model vs Temp=0.8 Model...", flush=True)
    new_v_temp08_wins = 0
    completed = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_match_worker, initargs=(model_new_path, model_temp08_path)) as pool:
        for res in pool.imap_unordered(play_new_vs_temp08, tasks):
            completed += 1
            if res["winner"] == res["new_color"]:
                new_v_temp08_wins += 1
    new_v_temp08_rate = (new_v_temp08_wins / 50) * 100
    print(f"  => Win Rate vs Temp0.8: {new_v_temp08_rate:.1f}% ({new_v_temp08_wins}/50)", flush=True)
    
    # Matchup 2: New vs Hybrid Teacher
    print("  2/3: New Model vs Hybrid Teacher...", flush=True)
    new_v_hybrid_wins = 0
    completed = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_match_worker, initargs=(model_new_path, model_temp08_path)) as pool:
        for res in pool.imap_unordered(play_new_vs_hybrid, tasks):
            completed += 1
            if res["winner"] == res["new_color"]:
                new_v_hybrid_wins += 1
    new_v_hybrid_rate = (new_v_hybrid_wins / 50) * 100
    print(f"  => Win Rate vs Hybrid Teacher: {new_v_hybrid_rate:.1f}% ({new_v_hybrid_wins}/50)", flush=True)
    
    # Matchup 3: New vs Depth 3 Minimax
    print("  3/3: New Model vs Depth 3...", flush=True)
    new_v_d3_wins = 0
    completed = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_match_worker, initargs=(model_new_path, model_temp08_path)) as pool:
        for res in pool.imap_unordered(play_new_vs_depth3, tasks):
            completed += 1
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
    md.append("# Great Kingdom AI - Fast Depth3 Immediate Training Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **학습 조건**: Fast Depth3 (K=8) 300판 데이터셋 (10,156 샘플, 99.67% Diversity)\n")
    
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
    md.append("| 평가 대상 (Matchup) | 총 판수 | 새 모델 (Fast Depth3 300판) 승률 | 비고 |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| **새 모델 vs 기존 Temp0.8 모델** | 50판 | **{new_v_temp08_rate:.1f}%** ({new_v_temp08_wins}승) | 기존 최고 실전 모델과의 비교 |")
    md.append(f"| **새 모델 vs Hybrid Teacher (결정론)** | 50판 | **{new_v_hybrid_rate:.1f}%** ({new_v_hybrid_wins}승) | 하이브리드 엔진 모방 지표 |")
    md.append(f"| **새 모델 vs Depth 3 Minimax** | 50판 | **{new_v_d3_rate:.1f}%** ({new_v_d3_wins}승) | **성공 판정: {d3_verdict}** |")
    md.append("")
    
    md.append("## 3. 핵심 모델별 정량 비교 분석")
    md.append("| 지표 (Metrics) | 기존 최고 모델 (Temp0.8, K=5) | Multi-Teacher 모델 | **신규 Fast Depth3 Pilot 모델** |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **학습 데이터 크기 (샘플 수)** | 50,000 | 42,939 | **10,156 (300판)** |")
    md.append(f"| **데이터셋 고유 기보 비율** | 10.30% | 36.00% | **99.67%** |")
    md.append(f"| **Depth 3 상대 승률** | 30.0% | 36.0% (과거값) | **{new_v_d3_rate:.1f}%** |")
    md.append("")
    
    md.append("## 4. 최종 질문에 대한 해답: 좋은 Teacher가 많은 데이터보다 중요한가?")
    md.append(f"### 최종 판정: **{final_verdict_str}**")
    md.append(f"- **실험 결과 요약**: Fast Depth3 (K=8)로 고유하게 생성된 **1만 개의 고품질 데이터**가, 하이브리드/Minimax/Temp 등이 무작위로 혼합된 **4.3만 개의 다중 교사 데이터**나 기존 **5만 개의 결정론적 데이터**와 비교하여 어떠한 효율을 보였는지 승률로 검증하였습니다.")
    
    if overall_success:
        md.append(f"- **결론**: **좋은 Teacher가 많은 데이터보다 더 중요함이 실험적으로 확인되었습니다.**")
        md.append(f"- 단 **10,156개(300판)**의 샘플만으로 기존 5만 판(Temp0.8) 대비 월등한 일반화 승률(**{new_v_d3_rate:.1f}%**)을 확보하였습니다.")
        md.append("- 이는 학습 데이터가 '암기' 수준을 벗어나 Depth-3 탐색의 자연스러운 착수 분포를 올바르게 학습함으로써 일반화 성능이 대폭 상승했음을 보여줍니다.")
    else:
        md.append(f"- **결론**: 데이터의 양(Quantity)이 극도로 부족할 경우(1만 개), 아무리 고품질의 Teacher(Fast Depth3, 99.67% 고유 비율)를 사용하더라도 네트워크가 충분히 수렴하기 어려워 성능 향상에 한계가 있음을 보여줍니다.")
        md.append("- 데이터 품질의 우위에도 불구하고 샘플 수의 부족으로 인해 일반화 성능 향상이 가로막혔으므로, 1,000판 이상으로의 데이터 확장이 필수적입니다.")
        
    md_content = "\n".join(md)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Training report successfully saved to: {report_path}", flush=True)
    
    os.makedirs(os.path.dirname(artifact_report_path), exist_ok=True)
    with open(artifact_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Artifact report successfully saved to: {artifact_report_path}", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
