import io
import os
import sys
import time
import random
import multiprocessing
import numpy as np
import torch
from contextlib import redirect_stdout

# Add project path to sys.path
sys.path.append(r"C:\Users\User\source\repos\greatkingdomAI")

from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from ai.minimax import find_best_move, clear_transposition_table, reset_stats, get_legal_moves
from model_v2 import PolicyNetworkV2
from ai.hybrid import find_hybrid_move, board_to_tensor, get_move_idx, get_move_from_idx

_worker_model = None

def init_worker(model_path):
    global _worker_model
    device = torch.device("cpu")
    _worker_model = PolicyNetworkV2().to(device)
    if os.path.exists(model_path):
        _worker_model.load_state_dict(torch.load(model_path, map_location=device))
    _worker_model.eval()

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

def play_one_selfplay_multi_teacher(args):
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
    
    # ----------------- Step 1: Select Teacher for this game -----------------
    # 1. Hybrid Deterministic (40%)
    # 2. Hybrid Temp 0.8, Top-k 5 (30%)
    # 3. Depth-2 Minimax (20%)
    # 4. Policy V2 (Generalized) (10%)
    teachers = ["hybrid_det", "hybrid_temp08", "depth2", "policy_v2"]
    weights = [0.4, 0.3, 0.2, 0.1]
    selected_teacher = random.choices(teachers, weights=weights, k=1)[0]
    
    while not game.game_over and move_count < max_moves:
        current_player = game.current_player
        
        # Save state tensor BEFORE the move
        state_tensor = board_to_tensor(game.board, current_player)
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if selected_teacher == "hybrid_det":
                move = find_hybrid_move(game, _worker_model, device, temperature=None)
            elif selected_teacher == "hybrid_temp08":
                move = find_hybrid_move(game, _worker_model, device, temperature=0.8, top_k=5)
            elif selected_teacher == "depth2":
                f = io.StringIO()
                with redirect_stdout(f):
                    move = find_best_move(game, depth=2)
            elif selected_teacher == "policy_v2":
                move = get_pure_policy_move(game, _worker_model, device)
            else:
                move = "pass"
            
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
        "teacher": selected_teacher,
        "history": history
    }

def run_selfplay_multi_teacher(num_games, model_path, output_path):
    print(f"--- Generating {num_games} Games with Multi-Teacher Self-Play ---", flush=True)
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
    
    teacher_counts = {"hybrid_det": 0, "hybrid_temp08": 0, "depth2": 0, "policy_v2": 0}
    teacher_wins = {"hybrid_det": 0, "hybrid_temp08": 0, "depth2": 0, "policy_v2": 0}
    teacher_total_moves = {"hybrid_det": 0, "hybrid_temp08": 0, "depth2": 0, "policy_v2": 0}
    
    pool_args = [(i, model_path) for i in range(1, num_games + 1)]
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
        for res in pool.imap_unordered(play_one_selfplay_multi_teacher, pool_args):
            completed += 1
            game_idx = res["game_idx"]
            winner = res["winner"]
            moves = res["moves"]
            termination = res["termination"]
            game_history = res["history"]
            teacher = res["teacher"]
            
            # Statistics per teacher
            teacher_counts[teacher] += 1
            teacher_total_moves[teacher] += moves
            if winner in [BLUE, ORANGE]:
                teacher_wins[teacher] += 1 
                
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
        "orange_win_pct": (orange_wins / num_games) * 100,
        "draw_pct": (draws / num_games) * 100,
        "teacher_counts": teacher_counts,
        "teacher_wins": teacher_wins,
        "teacher_total_moves": teacher_total_moves
    }

def main():
    print("=================================================================")
    print("GREAT KINGDOM AI - MULTI-TEACHER DIVERSITY EXPERIMENT")
    print("=================================================================")
    
    model_5000_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_5000.pth"
    output_npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_multi_teacher_1000.npz"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\multi_teacher_diversity_report.md"
    
    # Generate 1000 Games
    stats = run_selfplay_multi_teacher(1000, model_5000_path, output_npz_path)
    
    unique_pct = stats["unique_ratio"] * 100
    
    # Success evaluation for Uniqueness
    if unique_pct >= 20.0:
        verdict = "매우 성공 (20% 이상)"
    elif unique_pct >= 15.0:
        verdict = "성공 (15% 이상)"
    else:
        verdict = "실패 (15% 미만)"
        
    # Write report
    md = []
    md.append("# Great Kingdom AI - Multi-Teacher Diversity Experiment Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **검증 조건**: Multi-Teacher 혼합 자가대국 (1,000판)\n")
    
    md.append("## 1. 신규 데이터셋 다양성 결과")
    md.append(f"* **고유 기보 비율**: **{unique_pct:.2f}%** — **판정: {verdict}**")
    md.append(f"* **중복 기보 비율**: **{stats['duplicate_ratio']*100:.2f}%**")
    md.append(f"* **총 게임 수**: {stats['games']} 판")
    md.append(f"* **총 샘플 수**: {stats['samples']:,} 샘플")
    md.append(f"* **평균 게임 길이**: {stats['avg_moves']:.1f} 수")
    md.append(f"* **오프닝 다양성 (첫수 분포)**: {stats['opening_desc']}")
    md.append(f"* **승률 분포**: BLUE {stats['blue_win_pct']:.1f}% / ORANGE {stats['orange_win_pct']:.1f}% (무승부 {stats['draw_pct']:.1f}%)\n")
    
    md.append("## 2. Teacher별 선택 및 플레이 통계")
    md.append("| Teacher 종류 | 목표 확률 | 실제 배정 판수 | 비결정 판수 (승리 판수) | 평균 수순 |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for t in ["hybrid_det", "hybrid_temp08", "depth2", "policy_v2"]:
        count = stats["teacher_counts"][t]
        wins = stats["teacher_wins"][t]
        avg_t_moves = stats["teacher_total_moves"][t] / max(1, count)
        target_pct = {"hybrid_det": "40%", "hybrid_temp08": "30%", "depth2": "20%", "policy_v2": "10%"}[t]
        md.append(f"| **{t}** | {target_pct} | {count}판 ({count/10:.1f}%) | {wins}판 | {avg_t_moves:.1f} 수 |")
    md.append("")
    
    md.append("## 3. 데이터셋 다양성 변화 정량 비교")
    md.append("| 실험 조건 (Dataset Phase) | 고유 기보 비율 | 중복 기보 비율 | 비고 |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| **A) 결정론적 단일 Teacher** | 1.64% | 98.36% | 암기형 과적합 유발 |")
    md.append(f"| **B) Temp=0.8 + Top-k=5 (초반 10수)** | 10.30% | 89.70% | 과적합 우회 지점 |")
    md.append(f"| **C) Temp=0.8 + Top-k=5 (전체 게임)** | 10.60% | 89.40% | 전체 샘플링으로도 다양성 한계 |")
    md.append(f"| **D) Multi-Teacher 혼합 (신규)** | **{unique_pct:.2f}%** | {stats['duplicate_ratio']*100:.2f}% | **최종 판정: {verdict}** |")
    md.append("")
    
    md.append("## 4. 종합 분석 및 결론")
    if unique_pct >= 15.0:
        md.append("### 최종 결론: **성공 (다중 교사 전략 유효성 입증)**")
        md.append("- 단일 하이브리드 교사의 가중치나 온도를 조절하는 것보다, **원천이 다른 다중 교사(Minimax, Hybrid, Pure Policy)를 배정하는 전략이 데이터셋 다양성 확보에 매우 효과적임**이 입증되었습니다.")
        md.append("- 고유 기보 비율이 기존 단일 탐색 방식(10.60%) 대비 유의미한 수준으로 상승하여 **성공 기준(15% 이상)을 충족**했습니다.")
        md.append("- 이후 이 다양성 데이터셋을 기반으로 정책 네트워크를 재학습하면, 암기 편향을 대폭 극복하고 실전 일반화 성능(vs Depth-3 승률)을 향상시킬 수 있을 것입니다.")
    else:
        md.append("### 최종 결론: **실패 (다양성 추가 확보 방법론 모색 필요)**")
        md.append("- 다양한 교사를 섞었음에도 고유 기보 비율이 목표인 15%에 도달하지 못했습니다. 오프닝 초반 무작위 노이즈 추가 등 추가적인 교란 조치가 필요합니다.")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Report successfully saved to: {report_path}", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
