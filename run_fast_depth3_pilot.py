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
from ai.minimax import alphabeta, clear_transposition_table, reset_stats, get_legal_moves
from model_v2 import PolicyNetworkV2
from ai.hybrid import board_to_tensor, get_move_idx, get_move_from_idx

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
        
    # 1. Policy Network Inference
    state_np = board_to_tensor(game_state.board, curr_player)
    state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = policy_model(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # 2. Map legal moves to probabilities
    move_probs = []
    for m in legal_moves:
        idx = get_move_idx(m)
        prob = probs[idx]
        move_probs.append((m, prob))
        
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Select Top-K candidates
    top_candidates = [m for m, _ in move_probs[:k]]
    
    # 4. Perform Depth-3 search only on top K candidates
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
            
        # Root 1 move + Child Depth-2 search = Depth-3 search
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
        
        # Save state tensor BEFORE the move
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

def run_selfplay_fast_depth3(num_games, model_path, output_path):
    print(f"--- Generating {num_games} Pilot Games with Fast Depth3 (K=8) ---", flush=True)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    print(f"Using {num_processes} parallel processes.", flush=True)
    
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
                
            if completed % 50 == 0 or completed == num_games:
                elapsed = time.time() - start_time
                print(f"  [Progress {completed:03d}/{num_games:03d}] Elapsed: {elapsed:.1f}s | Avg Moves: {total_moves/completed:.1f}", flush=True)
                
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
        "draw_pct": (draws / num_games) * 100
    }

def main():
    print("=================================================================")
    print("GREAT KINGDOM AI - FAST DEPTH3 PILOT RUN (300 GAMES)")
    print("=================================================================")
    
    model_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_new_temp.pth"
    output_npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_fast_depth3_pilot_300.npz"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\fast_depth3_pilot_report.md"
    artifact_report_path = r"C:\Users\User\.gemini\antigravity-cli\brain\505d76c0-2e72-46fd-bbbf-aa02a413b645\fast_depth3_pilot_report.md"
    
    # Generate 300 Games
    stats = run_selfplay_fast_depth3(300, model_path, output_npz_path)
    
    unique_pct = stats["unique_ratio"] * 100
    
    # Success Evaluation
    if unique_pct >= 20.0:
        verdict = "매우 성공 (고유 기보 비율 20% 이상) -> 1000판 즉시 진행 권장"
    elif unique_pct >= 15.0:
        verdict = "성공 (고유 기보 비율 15% 이상) -> 1000판 진행 권장"
    elif unique_pct >= 10.0:
        verdict = "분석 대기 (고유 기보 비율 10~15%) -> 추가 검토 필요"
    else:
        verdict = "실패 및 중단 (고유 기보 비율 10% 미만) -> Fast Depth3 Teacher 부적합"
        
    # Write report
    md = []
    md.append("# Great Kingdom AI - Fast Depth3 Pilot Run Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **검증 조건**: Fast Depth3 (K=8) Self-Play (300판)\n")
    
    md.append("## 1. 데이터셋 다양성 결과")
    md.append(f"* **고유 기보 비율**: **{unique_pct:.2f}%** — **판정: {verdict}**")
    md.append(f"* **중복 기보 비율**: **{stats['duplicate_ratio']*100:.2f}%**")
    md.append(f"* **총 게임 수**: {stats['games']} 판")
    md.append(f"* **총 샘플 수**: {stats['samples']:,} 샘플")
    md.append(f"* **평균 게임 길이**: {stats['avg_moves']:.1f} 수")
    md.append(f"* **오프닝 다양성 (첫수 분포)**: {stats['opening_desc']}")
    md.append(f"* **승률 분포**: BLUE {stats['blue_win_pct']:.1f}% / ORANGE {stats['orange_win_pct']:.1f}% (무승부 {stats['draw_pct']:.1f}%)\n")
    
    md.append("## 2. 다양성 정량 비교")
    md.append("| 실험 조건 (Dataset Phase) | 고유 기보 비율 | 중복 기보 비율 | 비고 |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| **A) 결정론적 단일 Teacher** | 1.64% | 98.36% | 암기형 과적합 유발 |")
    md.append(f"| **B) Temp=0.8 + Top-k=5 (초반 10수)** | 10.30% | 89.70% | 과적합 우회 지점 |")
    md.append(f"| **C) Temp=0.8 + Top-k=5 (전체 게임)** | 10.60% | 89.40% | 전체 샘플링으로도 다양성 한계 |")
    md.append(f"| **D) Multi-Teacher 혼합** | 36.00% | 64.00% | 다양성 성공 (단, 지도 노이즈 존재) |")
    md.append(f"| **E) Fast Depth3 Pilot (신규)** | **{unique_pct:.2f}%** | {stats['duplicate_ratio']*100:.2f}% | **최종 판정: {verdict}** |")
    md.append("")
    
    md.append("## 3. 종합 분석 및 제안")
    if unique_pct >= 15.0:
        md.append("### 최종 결론: **SUCCESS (Fast Depth3 Teacher 도입 승인)**")
        md.append("- Fast Depth3 (K=8) 단일 교사만을 사용했음에도, 탐색 및 포석의 자연스러운 다양성에 기인하여 목표치(15%)를 달성하였습니다.")
        md.append("- Multi-Teacher의 지도 노이즈 문제를 해결하고 일관된 고품질(Depth-3 급)의 학습 신호를 정책 네트워크에 주입할 수 있습니다.")
    elif unique_pct >= 10.0:
        md.append("### 최종 결론: **PENDING (추가 분석 및 조정 필요)**")
        md.append("- 고유 기보 비율이 10~15% 수준으로 경계선에 머물러 있습니다. Dirichlet Noise 적용 또는 오프닝 단계에서의 무작위성 확장이 필요한지 검토합니다.")
    else:
        md.append("### 최종 결론: **ABORT (Fast Depth3 Teacher 부적합)**")
        md.append("- 고유 기보 비율이 10% 미만으로, 데이터의 중복성이 너무 높습니다. 단일 Fast Depth3 탐색기는 암기 편향을 피할 수 없습니다.")
        
    md_content = "\n".join(md)
    
    # Save to workspace
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Report successfully saved to: {report_path}", flush=True)
    
    # Save to artifacts directory
    os.makedirs(os.path.dirname(artifact_report_path), exist_ok=True)
    with open(artifact_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Artifact report successfully saved to: {artifact_report_path}", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
