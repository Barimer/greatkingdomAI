import os
import sys
import time
import numpy as np

def main():
    npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_dataset_v1.npz"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\selfplay_1000_games_report.md"
    
    print("=== Analyzing Self Play Dataset ===")
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} does not exist!")
        sys.exit(1)
        
    start_time = time.time()
    
    # 1. Load data & Check file corruption
    try:
        data = np.load(npz_path)
        states = data["states"]
        actions = data["actions"]
        players = data["players"]
        results = data["results"]
        game_ids = data["game_ids"]
        terminations = data["terminations"]
        file_corrupted = False
    except Exception as e:
        print(f"Error loading npz file: {e}")
        file_corrupted = True
        sys.exit(1)
        
    file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    
    # 2. Reconstruct individual games
    unique_game_ids = np.unique(game_ids)
    num_games = len(unique_game_ids)
    
    games = {}
    for g_id in unique_game_ids:
        # Find indices of moves for this game
        move_indices = np.where(game_ids == g_id)[0]
        # Sort indices to ensure order
        move_indices = np.sort(move_indices)
        
        game_moves = []
        for idx in move_indices:
            act = actions[idx]
            game_moves.append(tuple(act))
            
        games[g_id] = {
            "moves": tuple(game_moves),
            "winner": results[move_indices[0]] if len(move_indices) > 0 else None,
            "termination": terminations[g_id - 1] if g_id - 1 < len(terminations) else "UNKNOWN"
        }
        
    # 3. Calculate Duplicate Games
    game_sequences = [g["moves"] for g in games.values()]
    unique_sequences = set(game_sequences)
    num_unique_games = len(unique_sequences)
    duplicate_ratio = (num_games - num_unique_games) / num_games * 100
    
    # 4. Calculate Opening Diversity (distribution of first moves)
    first_moves = []
    for g_id, g_data in games.items():
        if len(g_data["moves"]) > 0:
            first_moves.append(g_data["moves"][0])
            
    first_move_counts = {}
    for mv in first_moves:
        first_move_counts[mv] = first_move_counts.get(mv, 0) + 1
        
    opening_diversity = len(first_move_counts)
    
    # Sort first moves by count descending
    sorted_first_moves = sorted(first_move_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 5. Calculate Win Rate Distribution
    blue_wins = sum(1 for g in games.values() if g["winner"] == 1)
    orange_wins = sum(1 for g in games.values() if g["winner"] == 2)
    
    # 6. Calculate Average Moves
    game_lengths = [len(g["moves"]) for g in games.values()]
    avg_moves = np.mean(game_lengths) if game_lengths else 0
    min_moves = np.min(game_lengths) if game_lengths else 0
    max_moves = np.max(game_lengths) if game_lengths else 0
    
    # 7. Calculate Termination Types
    term_counts = {}
    for g in games.values():
        term = g["termination"]
        term_counts[term] = term_counts.get(term, 0) + 1
        
    # 8. Memory Usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        mem_mb = 0.0
        
    # Generate Report Markdown
    md = []
    md.append("# Great Kingdom AI - Self Play 1000 Games Verification Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **검증 대상 파일**: `data/selfplay_dataset_v1.npz` ({file_size_mb:.2f} MB)\n")
    
    md.append("## 1. 생성 및 자원 통계 (Generation & Resource Statistics)")
    md.append(f"* **총 시뮬레이션 대국 수**: {num_games} 판")
    md.append(f"* **총 누적 샘플 수 (Moves)**: {len(states)} 개")
    md.append(f"* **파일 정상 여부**: {'정상 (NORMAL)' if not file_corrupted else '손상됨 (CORRUPTED)'}")
    md.append(f"* **파일 저장 용량**: {file_size_mb:.2f} MB")
    md.append(f"* **분석 프로세스 메모리 사용량 (RSS)**: {mem_mb:.2f} MB\n")
    
    md.append("## 2. 게임 길이 및 승률 분포 (Game Length & Win Rate)")
    md.append(f"* **BLUE 승률**: {blue_wins/num_games*100:.1f}% ({blue_wins}/{num_games})")
    md.append(f"* **ORANGE 승률**: {orange_wins/num_games*100:.1f}% ({orange_wins}/{num_games})")
    md.append(f"* **평균 수순 (Average Moves)**: {avg_moves:.1f} 수")
    md.append(f"* **최소 수순**: {min_moves} 수")
    md.append(f"* **최대 수순**: {max_moves} 수\n")
    
    md.append("## 3. 종료 유형 (Termination Types)")
    for term, count in term_counts.items():
        md.append(f"* **{term}**: {count/num_games*100:.1f}% ({count}판)")
    md.append("")
    
    md.append("## 4. 기보 중복도 및 오프닝 다양성 (Game Duplication & Opening Diversity)")
    md.append(f"* **고유 기보 수**: {num_unique_games} / {num_games} 판")
    md.append(f"* **중복 기보 비율**: **{duplicate_ratio:.1f}%**")
    md.append(f"* **첫 수(오프닝) 고유 좌표 종류**: {opening_diversity} 개 (총 81개 좌표 중)")
    md.append(f"* **오프닝 첫 수 분포 (Top 5)**:")
    for mv, count in sorted_first_moves[:5]:
        md.append(f"  * 좌표 `{mv}`: {count}회 ({count/num_games*100:.1f}%)")
    md.append("")
    
    md.append("## 5. 파이프라인 검증 판정")
    # 조건 만족 판정
    reasons = []
    passed = True
    
    if file_corrupted:
        passed = False
        reasons.append("파일이 손상되었습니다.")
    if duplicate_ratio > 10.0:
        passed = False
        reasons.append(f"중복 기보 비율이 너무 높습니다 ({duplicate_ratio:.1f}% > 10%).")
    if abs(blue_wins - orange_wins) / num_games > 0.4:
        passed = False
        reasons.append(f"승률 편향이 심합니다 (BLUE {blue_wins} vs ORANGE {orange_wins}).")
    if len(reasons) == 0:
        status_str = "PASS (검증 통과)"
        reasons.append("모든 조건(파일 무결성, 승률 분산, 중복도 제한 등)을 충족하였습니다. 1000판 대규모 생성으로 진입이 승인됩니다.")
    else:
        status_str = "FAIL (검증 실패)"
        
    md.append(f"### 최종 결과: **{status_str}**\n")
    md.append("### 판정 근거:")
    for r in reasons:
        md.append(f"1. {r}")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Analysis complete. Report generated at: {report_path}")

if __name__ == "__main__":
    main()
