import sys
import os
import time
import random
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add project path to sys.path
sys.path.append(r"C:\Users\User\source\repos\greatkingdomAI")

from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from ai.minimax import find_best_move, clear_transposition_table, reset_stats, get_legal_moves, copy_game_state
from model import PolicyNetwork
from model_v2 import PolicyNetworkV2
from ai.hybrid import board_to_tensor, get_move_idx, get_move_from_idx, find_hybrid_move

# Shared variables for workers
_learner_model = None
_v1_model = None

def init_league_worker(learner_path, v1_path):
    global _learner_model, _v1_model
    device = torch.device("cpu")
    
    if learner_path and os.path.exists(learner_path):
        _learner_model = PolicyNetworkV2().to(device)
        _learner_model.load_state_dict(torch.load(learner_path, map_location=device))
        _learner_model.eval()
        
    if v1_path and os.path.exists(v1_path):
        _v1_model = PolicyNetwork().to(device)
        _v1_model.load_state_dict(torch.load(v1_path, map_location=device))
        _v1_model.eval()

def get_pure_v2_move(game, model, device):
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

def get_pure_v1_move(game, model, device):
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

def play_league_game(args):
    game_idx, learner_color, opponent_type = args
    global _learner_model, _v1_model
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    device = torch.device("cpu")
    move_count = 0
    max_moves = 120
    
    # Store history for both players
    learner_history = []  # list of (state_np, move_coord)
    opponent_history = []
    
    while not game.game_over and move_count < max_moves:
        curr = game.current_player
        
        # Collect current state representation before moving
        state_np = board_to_tensor(game.board, curr)
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if curr == learner_color:
                move = get_pure_v2_move(game, _learner_model, device)
            else:
                if opponent_type == "policy_v2":
                    move = get_pure_v2_move(game, _learner_model, device)
                elif opponent_type == "policy_v1":
                    move = get_pure_v1_move(game, _v1_model, device)
                elif opponent_type == "depth3":
                    import io
                    from contextlib import redirect_stdout
                    f = io.StringIO()
                    with redirect_stdout(f):
                        move = find_best_move(game, depth=3)
                elif opponent_type == "hybrid":
                    move = find_hybrid_move(game, _learner_model, device, temperature=None)
                else:
                    raise ValueError(f"Unknown opponent: {opponent_type}")
                    
        # Record history
        # move is either (r, c) or "pass"
        move_coord = [-1, -1] if move == "pass" else [move[0], move[1]]
        
        if curr == learner_color:
            learner_history.append((state_np, move_coord))
        else:
            opponent_history.append((state_np, move_coord))
            
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                game.play_pass()
        move_count += 1
        
    winner = game.winner if game.winner is not None else game.check_winner()
    
    # Winner-only data extraction
    winner_states = []
    winner_actions = []
    
    if winner == learner_color:
        for s, a in learner_history:
            winner_states.append(s)
            winner_actions.append(a)
        learner_won = True
    else:
        for s, a in opponent_history:
            winner_states.append(s)
            winner_actions.append(a)
        learner_won = False
        
    return {
        "game_idx": game_idx,
        "opponent_type": opponent_type,
        "learner_won": learner_won,
        "winner": winner,
        "moves": move_count,
        "states": winner_states,
        "actions": winner_actions
    }

def play_validation_match(args):
    game_idx, league_color, opponent_type, league_model_path = args
    clear_transposition_table()
    reset_stats()
    
    # Load league model locally for validation
    device = torch.device("cpu")
    league_model = PolicyNetworkV2().to(device)
    league_model.load_state_dict(torch.load(league_model_path, map_location=device))
    league_model.eval()
    
    # Load comparison models (policy_rl_v2_e3 and policy_model_v1)
    learner_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_rl_v2_e3.pt"
    v2_model = PolicyNetworkV2().to(device)
    v2_model.load_state_dict(torch.load(learner_path, map_location=device))
    v2_model.eval()
    
    game = GameState()
    move_count = 0
    max_moves = 120
    start_time = time.time()
    
    while not game.game_over and move_count < max_moves:
        curr = game.current_player
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if curr == league_color:
                move = get_pure_v2_move(game, league_model, device)
            else:
                if opponent_type == "policy_v2":
                    move = get_pure_v2_move(game, v2_model, device)
                elif opponent_type == "depth3":
                    import io
                    from contextlib import redirect_stdout
                    f = io.StringIO()
                    with redirect_stdout(f):
                        move = find_best_move(game, depth=3)
                elif opponent_type == "hybrid":
                    move = find_hybrid_move(game, v2_model, device, temperature=None)
                else:
                    raise ValueError(f"Unknown opponent: {opponent_type}")
                    
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                game.play_pass()
        move_count += 1
        
    winner = game.winner if game.winner is not None else game.check_winner()
    duration = time.time() - start_time
    
    is_win = (winner == league_color)
    is_draw = (winner is None)
    
    return {
        "game_idx": game_idx,
        "winner": winner,
        "is_win": is_win,
        "is_draw": is_draw,
        "moves": move_count,
        "duration": duration
    }

class LeagueDataset(Dataset):
    def __init__(self, states, actions, mode="train", split_ratio=0.9, seed=42):
        action_indices = np.array([get_move_idx(act) for act in actions], dtype=np.int64)
        num_samples = len(states)
        indices = np.arange(num_samples)
        
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * split_ratio)
        if mode == "train":
            self.subset_indices = indices[:split_idx]
        else:
            self.subset_indices = indices[split_idx:]
            
        self.states = states
        self.action_indices = action_indices
        self.mode = mode
        
    def __len__(self):
        return len(self.subset_indices)
        
    def __getitem__(self, idx):
        real_idx = self.subset_indices[idx]
        state = self.states[real_idx]
        action_idx = self.action_indices[real_idx]
        
        # Apply symmetries in training
        if self.mode == "train":
            rot = np.random.randint(0, 4)
            flip = np.random.randint(0, 2)
            state = np.rot90(state, rot, axes=(1, 2))
            
            if action_idx != 81:
                r, c = action_idx // 9, action_idx % 9
                r_temp, c_temp = r, c
                for _ in range(rot):
                    r_temp, c_temp = 8 - c_temp, r_temp
                r, c = r_temp, c_temp
                
                if flip:
                    state = np.flip(state, axis=2)
                    c = 8 - c
                action_idx = r * 9 + c
            else:
                if flip:
                    state = np.flip(state, axis=2)
            state = state.copy()
            
        return torch.tensor(state, dtype=torch.float32), torch.tensor(action_idx, dtype=torch.long)

def main():
    learner_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_rl_v2_e3.pt"
    v1_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v1.pth"
    
    num_games = 500
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    
    # 1. Sample opponents based on league proportions:
    # 30% policy_rl_v2_e3, 30% policy_model_v1, 20% Depth3, 20% Hybrid
    opponents_pool = (
        ["policy_v2"] * 150 +
        ["policy_v1"] * 150 +
        ["depth3"] * 100 +
        ["hybrid"] * 100
    )
    random.shuffle(opponents_pool) # Shuffle to randomize matchmaking
    
    tasks = []
    for i in range(1, num_games + 1):
        color = BLUE if i % 2 == 1 else ORANGE
        opponent = opponents_pool[i - 1]
        tasks.append((i, color, opponent))
        
    print("=========================================================")
    print("그레이트 킹덤 AI - 리그 학습 단계 (League Training v1) 시작")
    print("=========================================================")
    print(f"1. 리그 대국 생성 중 ({num_games} 판)...")
    
    all_states = []
    all_actions = []
    
    # Stats counters
    opponent_stats = {
        "policy_v2": {"wins": 0, "games": 0, "moves": []},
        "policy_v1": {"wins": 0, "games": 0, "moves": []},
        "depth3": {"wins": 0, "games": 0, "moves": []},
        "hybrid": {"wins": 0, "games": 0, "moves": []}
    }
    
    t0 = time.time()
    completed = 0
    with multiprocessing.Pool(processes=num_processes, initializer=init_league_worker, initargs=(learner_path, v1_path)) as pool:
        for res in pool.imap_unordered(play_league_game, tasks):
            completed += 1
            opp = res["opponent_type"]
            opponent_stats[opp]["games"] += 1
            opponent_stats[opp]["moves"].append(res["moves"])
            if res["learner_won"]:
                opponent_stats[opp]["wins"] += 1
                
            all_states.extend(res["states"])
            all_actions.extend(res["actions"])
            
            if completed % 50 == 0:
                print(f"    - 대국 진행률: {completed}/{num_games} 판 완료...", flush=True)
                
    print(f"  리그 대국 완료! (소요시간: {time.time()-t0:.1f}초)")
    print(f"  수집된 승리자 행동 샘플 수: {len(all_states)} 개")
    
    # Save raw dataset
    dataset_npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\league_training_dataset.npz"
    np.savez_compressed(
        dataset_npz_path,
        states=np.array(all_states, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.int8)
    )
    print(f"  데이터셋 저장 완료: {dataset_npz_path}")
    
    # 2. Fine-Tuning PolicyNetworkV2 on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n2. Fine-Tuning 진행 중... (Device: {device})")
    
    train_dataset = LeagueDataset(all_states, all_actions, mode="train", split_ratio=0.9)
    val_dataset = LeagueDataset(all_states, all_actions, mode="val", split_ratio=0.9)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    model = PolicyNetworkV2().to(device)
    model.load_state_dict(torch.load(learner_path, map_location=device))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01) # Fine-tuning learning rate
    
    epochs = [2, 3, 5]
    max_epoch = max(epochs)
    
    validation_accuracy = {}
    
    for ep in range(1, max_epoch + 1):
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
            train_loss += loss.item() * states.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation accuracy check
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for states, targets in val_loader:
                states = states.to(device)
                targets = targets.to(device)
                outputs = model(states)
                _, preds = outputs.topk(1, dim=1)
                val_correct += preds.eq(targets.view(-1, 1)).sum().item()
        val_acc = (val_correct / len(val_dataset)) * 100
        
        print(f"  Epoch {ep}/{max_epoch} | Train Loss: {train_loss:.4f} | Val Acc (Top-1): {val_acc:.2f}%")
        
        if ep in epochs:
            # Save checkpoints
            checkpoint_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_league_e{ep}.pt".format(ep=ep)
            torch.save(model.state_dict(), checkpoint_path)
            validation_accuracy[ep] = val_acc
            
    # Select best epoch based on Validation Accuracy
    best_epoch = max(validation_accuracy, key=validation_accuracy.get)
    best_model_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_league_e{ep}.pt".format(ep=best_epoch)
    print(f"\n최고 모델 선정: Epoch {best_epoch} (Val Acc: {validation_accuracy[best_epoch]:.2f}%)")
    
    # 3. Validation Matches (100 games each)
    num_validation_games = 100
    print(f"\n3. 리그 검증 대국 구동 중 (최고 모델 vs Opponents, 각 {num_validation_games}판)...")
    
    # Matchup 1: vs policy_rl_v2_e3
    print(f"  1/3: policy_league vs policy_rl_v2_e3...", flush=True)
    tasks_v2 = []
    for i in range(1, num_validation_games + 1):
        color = BLUE if i % 2 == 1 else ORANGE
        tasks_v2.append((i, color, "policy_v2", best_model_path))
    v2_wins = 0
    t_start = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        for res in pool.imap_unordered(play_validation_match, tasks_v2):
            if res["is_win"]:
                v2_wins += 1
    v2_rate = (v2_wins / num_validation_games) * 100
    print(f"    완료 (소요시간: {time.time()-t_start:.1f}초) | 승률: {v2_rate:.1f}% ({v2_wins}/{num_validation_games})", flush=True)
    
    # Matchup 2: vs Depth3 Minimax
    print(f"  2/3: policy_league vs Depth3 Minimax...", flush=True)
    tasks_d3 = []
    for i in range(1, num_validation_games + 1):
        color = BLUE if i % 2 == 1 else ORANGE
        tasks_d3.append((i, color, "depth3", best_model_path))
    d3_wins = 0
    t_start = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        for res in pool.imap_unordered(play_validation_match, tasks_d3):
            if res["is_win"]:
                d3_wins += 1
    d3_rate = (d3_wins / num_validation_games) * 100
    print(f"    완료 (소요시간: {time.time()-t_start:.1f}초) | 승률: {d3_rate:.1f}% ({d3_wins}/{num_validation_games})", flush=True)
    
    # Matchup 3: vs Hybrid AI
    print(f"  3/3: policy_league vs Hybrid AI...", flush=True)
    tasks_hybrid = []
    for i in range(1, num_validation_games + 1):
        color = BLUE if i % 2 == 1 else ORANGE
        tasks_hybrid.append((i, color, "hybrid", best_model_path))
    hybrid_wins = 0
    t_start = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        for res in pool.imap_unordered(play_validation_match, tasks_hybrid):
            if res["is_win"]:
                hybrid_wins += 1
    hybrid_rate = (hybrid_wins / num_validation_games) * 100
    print(f"    완료 (소요시간: {time.time()-t_start:.1f}초) | 승률: {hybrid_rate:.1f}% ({hybrid_wins}/{num_validation_games})", flush=True)
    
    # 4. Generate League Training Report
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\league_training_report.md"
    artifact_report_path = r"C:\Users\User\.gemini\antigravity-cli\brain\0b1bab1b-8dc5-4e68-aaaf-50b0888fa3eb\league_training_report.md"
    
    md = []
    md.append("# Great Kingdom AI - League Training v1 Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **리그 대국 수**: {num_games} 판 (승자 기보 추출)")
    md.append(f"* **최고 에포크 모델**: `policy_league_e{best_epoch}.pt` (Val Acc: {validation_accuracy[best_epoch]:.2f}%)\n")
    
    md.append("## 1. 리그 대국 생성 통계")
    md.append("| 상대 유형 (Opponent) | 대국 수 | 학습자(Learner) 승리 | 학습자 승률 | 평균 수순 |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for opp, stats in opponent_stats.items():
        opp_rate = (stats["wins"] / stats["games"]) * 100 if stats["games"] > 0 else 0.0
        opp_avg_moves = np.mean(stats["moves"]) if stats["moves"] else 0.0
        md.append(f"| **{opp}** | {stats['games']}판 | {stats['wins']}승 | {opp_rate:.1f}% | {opp_avg_moves:.1f} 수 |")
    md.append("")
    
    md.append("## 2. Fine-Tuning 체크포인트 성능 비교")
    md.append("| 모델 Checkpoint | Epoch | Validation Accuracy (Top-1) |")
    md.append("| :--- | :---: | :---: |")
    for ep in epochs:
        md.append(f"| `policy_league_e{ep}.pt` | {ep} | {validation_accuracy[ep]:.2f}% |")
    md.append("")
    
    md.append("## 3. 리그 챔피언 실전 대국 결과 요약 (100판씩)")
    md.append("| 대국 대상 (Opponent) | 리그 모델 승률 | 결과 판정 | 성공 기준 |")
    md.append("| :--- | :---: | :---: | :---: |")
    
    v2_verdict = "성공 (55% 이상)" if v2_rate >= 55.0 else "실패 (50% 이하)" if v2_rate <= 50.0 else "유지 (50% ~ 55%)"
    md.append(f"| **vs policy_rl_v2_e3** | {v2_rate:.1f}% ({v2_wins}/{num_validation_games}) | {v2_verdict} | 55% 이상 |")
    
    d3_verdict = "매우 성공 (60% 이상)" if d3_rate >= 60.0 else "성공 (55% 이상)" if d3_rate >= 55.0 else "실패 (52% 이하)" if d3_rate <= 52.0 else "보통 (52% ~ 55%)"
    md.append(f"| **vs Depth3 Minimax** | **{d3_rate:.1f}%** ({d3_wins}/{num_validation_games}) | **{d3_verdict}** | 55% 이상 |")
    
    hybrid_verdict = "우세 (50% 이상)" if hybrid_rate >= 50.0 else "열세 (50% 미만)"
    md.append(f"| **vs Hybrid AI** | {hybrid_rate:.1f}% ({hybrid_wins}/{num_validation_games}) | {hybrid_verdict} | - |")
    md.append("")
    
    md.append("## 4. 최종 핵심 질문에 대한 답변")
    md.append("### Q. League Training이 기존 Winner-Only RL보다 실제로 일반화 성능을 향상시키는가?")
    
    is_success = (v2_rate >= 55.0) and (d3_rate >= 55.0)
    
    if is_success:
        md.append("- **답변**: **네, 일반화 성능이 확실하게 향상되었습니다.**")
        md.append(f"- 기존 최고 모델(`policy_rl_v2_e3`)을 상대로 승률 **{v2_rate:.1f}%**를 기록하며 성공했고, 외부 벤치마크인 Depth3 Minimax 상대로도 승률 **{d3_rate:.1f}%**를 확보하여 성공 기준선(55%)을 넘어섰습니다.")
        md.append("- 특정 자가 대국 패턴에만 과적합되던 기존의 Self-Play 방식과 달리, 다양한 세대의 기보(V1, V2) 및 정밀한 규칙 기반 교사(Depth3, Hybrid)를 리그 형태로 결합하여 대국을 생성함으로써, 다양한 방어 및 포석 전술을 균형 있게 학습하였습니다. 이는 Self-Play Collapse를 극복하는 실질적인 대안입니다.")
    else:
        md.append("- **답변**: **일반화 성능의 향상이 미미하거나 성공 기준선에 도달하지 못했습니다.**")
        md.append("- 기존 최고 모델인 `policy_rl_v2_e3` 대비 확실한 우세를 확보하지 못했거나, 여전히 Depth 3 Minimax의 빈틈없는 탐색 전술에 밀리는 한계를 보였습니다.")
        md.append("- 따라서 현 단계의 모델은 기존 챔피언 가중치(`policy_rl_v2_e3.pt`)를 넘지 못해 기존 모델을 그대로 챔피언으로 유지합니다.")
        
    md_content = "\n".join(md)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nReport saved to: {report_path}")
    
    os.makedirs(os.path.dirname(artifact_report_path), exist_ok=True)
    with open(artifact_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Artifact report saved to: {artifact_report_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
