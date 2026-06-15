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
from ai.minimax import find_best_move, alphabeta, clear_transposition_table, reset_stats, get_legal_moves, STATS
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
def find_policy_guided_depth3_move(game_state, policy_model, device, k=5):
    if game_state.game_over:
        return "pass"
        
    curr_player = game_state.current_player
    legal_moves = get_legal_moves(game_state)
    if not legal_moves:
        return "pass"
        
    # 1. Policy Network 추론
    state_np = board_to_tensor(game_state.board, curr_player)
    state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = policy_model(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # 2. 합법수들의 확률 매핑 및 정렬
    move_probs = []
    for m in legal_moves:
        idx = get_move_idx(m)
        prob = probs[idx]
        move_probs.append((m, prob))
        
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    # 3. 상위 K개 후보 선택
    top_candidates = [m for m, _ in move_probs[:k]]
    
    # 4. 상위 K개 후보에 대해서만 Depth3 탐색 (자식은 Depth2)
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
            
        # Depth 3 탐색이므로, 루트 1수 play 후 남은 깊이는 2
        score = alphabeta(next_state, depth=2, alpha=alpha, beta=beta, maximizing_player=False, target_player=target_player)
        move_scores.append((move, score))
        
        alpha = max(alpha, score)
        
    if move_scores:
        move_scores.sort(key=lambda x: x[1], reverse=True)
        best_score = move_scores[0][1]
        best_candidates = [move for move, score in move_scores if abs(score - best_score) < 1e-7]
        return random.choice(best_candidates)
        
    return "pass"

# ----------------- Battle Simulation: Original vs Guided -----------------
def play_original_vs_guided(args):
    game_idx, guided_k, model_path, guided_color = args
    global _worker_model
    
    if _worker_model is None:
        init_worker(model_path)
        
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    device = torch.device("cpu")
    
    move_count = 0
    max_moves = 150
    game_start = time.time()
    
    # metrics tracking
    duration_orig = []
    duration_guided = []
    nodes_orig = []
    nodes_guided = []
    
    while not game.game_over and move_count < max_moves:
        curr_player = game.current_player
        
        if move_count == 0:
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            if curr_player == guided_color:
                # Guided Depth3
                reset_stats()
                t0 = time.time()
                move = find_policy_guided_depth3_move(game, _worker_model, device, k=guided_k)
                duration_guided.append(time.time() - t0)
                nodes_guided.append(STATS["nodes_visited"])
            else:
                # Original Depth3
                reset_stats()
                t0 = time.time()
                f = io.StringIO()
                with redirect_stdout(f):
                    move = find_best_move(game, depth=3)
                duration_orig.append(time.time() - t0)
                nodes_orig.append(STATS["nodes_visited"])
                
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
        "guided_color": guided_color,
        "moves": move_count,
        "avg_dur_orig": np.mean(duration_orig) if duration_orig else 0.0,
        "avg_dur_guided": np.mean(duration_guided) if duration_guided else 0.0,
        "avg_nodes_orig": np.mean(nodes_orig) if nodes_orig else 0.0,
        "avg_nodes_guided": np.mean(nodes_guided) if nodes_guided else 0.0,
        "duration": time.time() - game_start
    }

def run_experiment_for_k(guided_k, model_path, num_games=20):
    print(f"\n--- Running Experiment for K = {guided_k} ({num_games} Games) ---", flush=True)
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 1)
    
    # Balanced colors
    tasks = []
    for i in range(1, num_games + 1):
        guided_color = BLUE if i <= (num_games // 2) else ORANGE
        tasks.append((i, guided_k, model_path, guided_color))
        
    guided_wins = 0
    completed = 0
    
    dur_orig_list = []
    dur_guided_list = []
    nodes_orig_list = []
    nodes_guided_list = []
    total_game_times = []
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
        for res in pool.imap_unordered(play_original_vs_guided, tasks):
            completed += 1
            guided_color = res["guided_color"]
            winner = res["winner"]
            
            if winner == guided_color:
                guided_wins += 1
                
            dur_orig_list.append(res["avg_dur_orig"])
            dur_guided_list.append(res["avg_dur_guided"])
            nodes_orig_list.append(res["avg_nodes_orig"])
            nodes_guided_list.append(res["avg_nodes_guided"])
            total_game_times.append(res["duration"])
            
            if completed % 5 == 0 or completed == num_games:
                print(f"  [Progress {completed:02d}/{num_games:02d}] Guided Win Rate: {(guided_wins/completed)*100:.1f}% | Avg Guided Dur: {np.mean(dur_guided_list)*1000:.1f}ms", flush=True)
                
    win_rate = (guided_wins / num_games) * 100
    avg_dur_orig = np.mean(dur_orig_list) * 1000  # ms
    avg_dur_guided = np.mean(dur_guided_list) * 1000  # ms
    avg_nodes_orig = np.mean(nodes_orig_list)
    avg_nodes_guided = np.mean(nodes_guided_list)
    
    speedup = avg_dur_orig / max(1e-9, avg_dur_guided)
    node_reduction = (avg_nodes_orig - avg_nodes_guided) / max(1e-9, avg_nodes_orig) * 100
    
    return {
        "k": guided_k,
        "win_rate": win_rate,
        "avg_dur_orig": avg_dur_orig,
        "avg_dur_guided": avg_dur_guided,
        "avg_nodes_orig": avg_nodes_orig,
        "avg_nodes_guided": avg_nodes_guided,
        "speedup": speedup,
        "node_reduction": node_reduction,
        "total_time": np.sum(total_game_times)
    }

def main():
    print("=================================================================")
    print("GREAT KINGDOM AI - FAST DEPTH3 PROJECT BENCHMARKS")
    print("=================================================================")
    
    model_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v2_new_temp.pth"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\fast_depth3_report.md"
    
    start_time = time.time()
    
    # Run experiments for K = 5, 8, 10
    results = []
    for k in [5, 8, 10]:
        res = run_experiment_for_k(k, model_path, num_games=20)
        results.append(res)
        
    total_elapsed = time.time() - start_time
    print(f"\nAll experiments finished in {total_elapsed:.1f}s", flush=True)
    
    # Assessment & Write report
    md = []
    md.append("# Great Kingdom AI - Fast Depth3 Project Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **정책 네트워크**: 기존 최고 모델 (`policy_model_v2_new_temp.pth`)\n")
    
    md.append("## 1. 실험 결과 요약 (Benchmark Summary)")
    md.append("| 조건 (K 값) | 오리지널 Depth3 대비 승률 | 평균 추론 시간 (Guided) | 오리지널 추론 시간 | 속도 향상 배율 | 노드 수 감소율 | 최종 판정 |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    for r in results:
        win_pct = r["win_rate"]
        dur_orig = r["avg_dur_orig"]
        dur_guided = r["avg_dur_guided"]
        speedup = r["speedup"]
        reduction = r["node_reduction"]
        
        # Success check
        win_ok = win_pct >= 45.0  # 90% of 50% target
        speed_ok = speedup >= 3.0
        
        if win_ok and speed_ok:
            verdict = "성공 (SUCCESS)"
        elif win_ok:
            verdict = "속도 미달 (Low Speedup)"
        elif speed_ok:
            verdict = "승률 미달 (Low Winrate)"
        else:
            verdict = "실패 (FAIL)"
            
        md.append(f"| **K = {r['k']}** | {win_pct:.1f}% | {dur_guided:.1f} ms | {dur_orig:.1f} ms | **{speedup:.2f}배** | **{reduction:.1f}%** | {verdict} |")
    md.append("")
    
    md.append("## 2. 상세 성능 메트릭 (Detailed Metrics)")
    for r in results:
        md.append(f"### 조건 K = {r['k']} 상세 통계")
        md.append(f"* **평균 방문 노드 수 (Original)**: {r['avg_nodes_orig']:.1f} 노드")
        md.append(f"* **평균 방문 노드 수 (Guided)**: {r['avg_nodes_guided']:.1f} 노드")
        md.append(f"* **평균 노드 감소율**: **{r['node_reduction']:.2f}%**")
        md.append(f"* **평균 추론 시간 (Original)**: {r['avg_dur_orig']:.1f} ms")
        md.append(f"* **평균 추론 시간 (Guided)**: {r['avg_dur_guided']:.1f} ms")
        md.append(f"* **속도 개선 배율**: **{r['speedup']:.2f} 배 향상**")
        md.append(f"* **오리지널 대비 승률**: **{r['win_rate']:.1f}%** (목표 45.0% 이상)\n")
        
    md.append("## 3. 종합 분석 및 제안")
    
    # Find the best K
    best_r = None
    for r in results:
        if r["win_rate"] >= 45.0 and r["speedup"] >= 3.0:
            if best_r is None or r["win_rate"] > best_r["win_rate"]:
                best_r = r
                
    if best_r:
        md.append(f"### 최종 판정: **성공 (FAST DEPTH3 타당성 검증 완료)**")
        md.append(f"- **최적 조건**: **K = {best_r['k']}** (승률 {best_r['win_rate']:.1f}%, 속도 {best_r['speedup']:.2f}배 향상)")
        md.append(f"- 정책 네트워크로 합법수를 필터링하여 상위 {best_r['k']}개 후보군에 대해서만 Depth3 탐색을 실행한 결과, 탐색 노드 수를 **{best_r['node_reduction']:.1f}%** 감축시켰습니다.")
        md.append(f"- 이는 오리지널 Depth3의 기량을 **90% 이상(승률 {best_r['win_rate']:.1f}%) 유지하면서도, 데이터 생성 속도를 {best_r['speedup']:.2f}배 단축**시킬 수 있음을 입증합니다.")
        md.append("- 향후 자가 대국 생성기(Self-Play Teacher)에 이 **Fast Depth3 (Policy Guided Depth3, K = {best_r['k']})**를 도입하여 대규모 데이터셋(5,000판 이상)을 수집하는 차기 로드맵 집행을 강력히 권장합니다.")
    else:
        md.append(f"### 최종 판정: **실패 (추가 성능 튜닝 필요)**")
        md.append("- 승률과 속도 향상 조건을 동시에 충족하는 최적의 K값을 찾지 못했습니다. Alpha-Beta 탐색 노이즈나 정렬 가중치의 보완이 필요합니다.")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Fast Depth3 report successfully saved to: {report_path}", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
