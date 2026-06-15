import os
import sys
import time
import numpy as np

def main():
    npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_dataset_v1.npz"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\dataset_integrity_report.md"
    
    print("=== STARTING DATASET INTEGRITY VERIFICATION ===")
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} does not exist!")
        sys.exit(1)
        
    # 1. Load npz dataset
    try:
        dataset = np.load(npz_path)
        keys = list(dataset.files)
        print(f"Loaded keys: {keys}")
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        sys.exit(1)
        
    # 2. Extract arrays
    states = dataset["states"]
    actions = dataset["actions"]
    players = dataset["players"]
    results = dataset["results"]
    game_ids = dataset["game_ids"]
    terminations = dataset["terminations"]
    
    # 3. Calculate Memory Usage
    states_mem = states.nbytes / (1024 * 1024)
    actions_mem = actions.nbytes / (1024 * 1024)
    players_mem = players.nbytes / (1024 * 1024)
    results_mem = results.nbytes / (1024 * 1024)
    game_ids_mem = game_ids.nbytes / (1024 * 1024)
    terminations_mem = terminations.nbytes / (1024 * 1024)
    
    total_uncompressed_mem = (
        states_mem + actions_mem + players_mem + results_mem + game_ids_mem + terminations_mem
    )
    
    actual_file_size = os.path.getsize(npz_path) / (1024 * 1024)
    compression_ratio = total_uncompressed_mem / actual_file_size if actual_file_size > 0 else 0.0
    
    # 4. Verify Tensor Content (One-Hot-like cell representation verification)
    # Each coordinate (r, c) must belong to exactly one channel.
    # Therefore, sum across axis=1 (channels) for each cell must be exactly 1.
    channel_sums = np.sum(states, axis=1)
    is_one_hot = np.all(channel_sums == 1)
    
    # Statistics of cell contents
    total_cells = states.shape[0] * 9 * 9
    my_stones_count = np.sum(states[:, 0, :, :])
    opp_stones_count = np.sum(states[:, 1, :, :])
    neutral_castles_count = np.sum(states[:, 2, :, :])
    empty_cells_count = np.sum(states[:, 3, :, :])
    
    # 5. Action Validation
    # Actions should be coordinates in [0, 8] or [-1, -1] for pass.
    min_action = np.min(actions)
    max_action = np.max(actions)
    
    passes_count = sum(1 for a in actions if np.array_equal(a, [-1, -1]))
    board_moves_count = len(actions) - passes_count
    
    # Action histogram (distribution on 9x9 grid)
    action_grid_counts = np.zeros((9, 9), dtype=np.int32)
    for act in actions:
        if not np.array_equal(act, [-1, -1]):
            action_grid_counts[act[0], act[1]] += 1
            
    # 6. Result Validation
    # Results should be 1 (BLUE) or 2 (ORANGE)
    unique_results = np.unique(results)
    blue_wins = np.sum(results == 1)
    orange_wins = np.sum(results == 2)
    
    # 7. Sample Dump
    # First Sample
    first_state = states[0]
    first_action = actions[0]
    first_player = players[0]
    first_result = results[0]
    
    # 5 Random Samples
    np.random.seed(42)
    random_indices = np.random.choice(len(states), 5, replace=False)
    
    # 8. Final Judgment
    judgement = "PASS"
    reasons = []
    
    if len(states) == 0:
        judgement = "FAIL"
        reasons.append("Dataset is empty.")
    elif states.shape[1:] != (4, 9, 9):
        judgement = "FAIL"
        reasons.append(f"States shape is invalid: {states.shape[1:]} != (4, 9, 9)")
        
    if not is_one_hot:
        judgement = "FAIL"
        reasons.append("Tensor content validation failed (cells do not sum to 1 across channels).")
        
    if min_action < -1 or max_action > 8:
        judgement = "FAIL"
        reasons.append(f"Action coordinates out of bounds: [{min_action}, {max_action}]")
        
    for res_val in unique_results:
        if res_val not in (1, 2):
            judgement = "WARNING"
            reasons.append(f"Found unexpected winner value: {res_val} (expected 1 or 2).")
            
    if duplicate_ratio_check(game_ids, actions) > 10.0:
        judgement = "WARNING"
        reasons.append("Duplicate game ratio is above 10%.")
        
    if len(reasons) == 0:
        reasons.append("All checks passed successfully. The dataset is fully valid and ready for training.")
        
    # Generate report markdown
    md = []
    md.append("# Great Kingdom AI - Dataset Integrity Verification Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **검증 대상**: `data/selfplay_dataset_v1.npz`\n")
    
    md.append("## 1. Dataset Keys & Types")
    md.append("| Key | Shape | Dtype |")
    md.append("| :--- | :---: | :---: |")
    for k in keys:
        md.append(f"| `{k}` | {dataset[k].shape} | `{dataset[k].dtype}` |")
    md.append("")
    
    md.append("## 2. Memory Usage & Compression")
    md.append(f"* **states**: {states_mem:.4f} MB")
    md.append(f"* **actions**: {actions_mem:.4f} MB")
    md.append(f"* **players**: {players_mem:.4f} MB")
    md.append(f"* **results**: {results_mem:.4f} MB")
    md.append(f"* **game_ids**: {game_ids_mem:.4f} MB")
    md.append(f"* **terminations**: {terminations_mem:.4f} MB")
    md.append(f"* **총 예상 uncompressed 용량**: **{total_uncompressed_mem:.4f} MB**")
    md.append(f"* **실제 npz 저장 용량**: **{actual_file_size:.4f} MB**")
    md.append(f"* **압축률 (Compression Ratio)**: **{compression_ratio:.2f}x** (압축으로 인해 용량이 작게 나타난 것임)\n")
    
    md.append("## 3. Tensor Content Validation")
    md.append(f"* **각 좌표 채널 합이 1인지 여부 (One-hot Grid Validation)**: **{is_one_hot}**")
    md.append(f"* **셀 통계 (Cell Occupancy)**:")
    md.append(f"  * 내 돌 (Channel 0): {my_stones_count}개 ({my_stones_count/total_cells*100:.2f}%)")
    md.append(f"  * 상대 돌 (Channel 1): {opp_stones_count}개 ({opp_stones_count/total_cells*100:.2f}%)")
    md.append(f"  * 중립 성 (Channel 2): {neutral_castles_count}개 ({neutral_castles_count/total_cells*100:.2f}%)")
    md.append(f"  * 빈 칸 (Channel 3): {empty_cells_count}개 ({empty_cells_count/total_cells*100:.2f}%)")
    md.append(f"  * 총 합 검증: {my_stones_count + opp_stones_count + neutral_castles_count + empty_cells_count} / {total_cells} 개 ({'성공' if my_stones_count + opp_stones_count + neutral_castles_count + empty_cells_count == total_cells else '실패'})\n")
    
    md.append("## 4. Action Validation")
    md.append(f"* **Action 좌표 범위**: `[{min_action}, {max_action}]` (합법 범위 내 존재)")
    md.append(f"* **착수 구분**:")
    md.append(f"  * 바둑판 착수 (Board Moves): {board_moves_count}회 ({board_moves_count/len(actions)*100:.1f}%)")
    md.append(f"  * 패스 (Pass): {passes_count}회 ({passes_count/len(actions)*100:.1f}%)")
    md.append("* **바둑판 착수 9x9 히스토그램 (Action Grid Distribution)**:")
    md.append("```")
    # Draw a 9x9 grid text
    grid_lines = []
    for r in range(9):
        line_str = " ".join(f"{action_grid_counts[r, c]:3d}" for c in range(9))
        grid_lines.append(f"Row {r}: {line_str}")
    md.append("\n".join(grid_lines))
    md.append("```\n")
    
    md.append("## 5. Result Validation")
    md.append(f"* **결과값 종류**: {list(unique_results)}")
    md.append(f"* **BLUE 승리 샘플 수**: {blue_wins}개 ({blue_wins/len(results)*100:.1f}%)")
    md.append(f"* **ORANGE 승리 샘플 수**: {orange_wins}개 ({orange_wins/len(results)*100:.1f}%)\n")
    
    md.append("## 6. First Sample Dump")
    md.append("### states[0] (Channels: My, Opp, Neutral, Empty)")
    md.append("```")
    for ch in range(4):
        md.append(f"Channel {ch}:")
        for r in range(9):
            md.append(" ".join(str(int(first_state[ch, r, c])) for c in range(9)))
    md.append("```")
    md.append(f"* **actions[0]**: {list(first_action)}")
    md.append(f"* **players[0]**: {first_player} ({'BLUE' if first_player == 1 else 'ORANGE'})")
    md.append(f"* **results[0]**: {first_result} ({'BLUE' if first_result == 1 else 'ORANGE'})\n")
    
    md.append("## 7. Random 5 Samples Dump")
    for idx in random_indices:
        md.append(f"### Sample Index: {idx}")
        md.append(f"* **action**: {list(actions[idx])}")
        md.append(f"* **player**: {players[idx]} ({'BLUE' if players[idx] == 1 else 'ORANGE'})")
        md.append(f"* **result (winner)**: {results[idx]} ({'BLUE' if results[idx] == 1 else 'ORANGE'})")
        # Print active channel counts
        my_c = np.sum(states[idx, 0])
        opp_c = np.sum(states[idx, 1])
        neu_c = np.sum(states[idx, 2])
        emp_c = np.sum(states[idx, 3])
        md.append(f"* **Stones Count**: My={int(my_c)}, Opp={int(opp_c)}, Neutral={int(neu_c)}, Empty={int(emp_c)}")
    md.append("")
    
    md.append("## 8. 최종 판정 (Final Judgment)")
    md.append(f"### 판정 결과: **{judgement}**\n")
    md.append("### 상세 사유:")
    for idx, r in enumerate(reasons, 1):
        md.append(f"{idx}. {r}")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Integrity report written successfully to: {report_path}")

def duplicate_ratio_check(game_ids, actions):
    unique_game_ids = np.unique(game_ids)
    num_games = len(unique_game_ids)
    games = []
    for g_id in unique_game_ids:
        indices = np.where(game_ids == g_id)[0]
        indices = np.sort(indices)
        games.append(tuple(tuple(actions[idx]) for idx in indices))
    return (num_games - len(set(games))) / num_games * 100

if __name__ == "__main__":
    main()
