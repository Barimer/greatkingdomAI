import os
import json
import sys
import glob

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from engine.capture import get_group, get_liberties
from engine.territory import calculate_territory

def analyze_single_game(game_file):
    with open(game_file, "r", encoding="utf-8") as f:
        game_data = json.load(f)
        
    winner = game_data["winner"]
    win_reason = game_data.get("win_reason", "CAPTURE")
    moves = game_data["moves"]
    
    # Game State 복원하여 시뮬레이션 진행
    game = GameState()
    game.is_copy = True
    
    # 종료 직전(결정타 직전) 상태 분석을 위해 마지막 수 직전까지만 재현
    for idx, move in enumerate(moves[:-1]):
        if game.game_over:
            break
        if move == "pass":
            game.play_pass()
        else:
            game.play_move(move[0], move[1])
            
    board = game.board
    
    # 1. 종료 방식 매핑
    termination = win_reason
    if termination == "MAX_MOVES":
        termination = "TERRITORY"
        
    # 2. 최종 수순 번호
    final_move = len(moves)
    
    # 3. 종료 직전 돌 개수
    blue_stones = sum(1 for r in range(board.size) for c in range(board.size) if board.get(r, c) == BLUE)
    orange_stones = sum(1 for r in range(board.size) for c in range(board.size) if board.get(r, c) == ORANGE)
    
    # 4. 자유도 (Liberties) 계산
    # 그레이트 킹덤은 단 1개의 그룹만 포위당해도 패배하므로,
    # 각 플레이어가 가진 돌 그룹들 중 '가장 위태로운(자유도가 최소인) 그룹의 자유도'를 구합니다.
    blue_group_libs = []
    orange_group_libs = []
    checked = set()
    
    for r in range(board.size):
        for c in range(board.size):
            stone = board.get(r, c)
            if stone in (BLUE, ORANGE) and (r, c) not in checked:
                group = get_group(board, r, c)
                checked.update(group)
                libs = get_liberties(board, group)
                lib_count = len(libs)
                if stone == BLUE:
                    blue_group_libs.append(lib_count)
                else:
                    orange_group_libs.append(lib_count)
                    
    blue_liberties = min(blue_group_libs) if blue_group_libs else 8
    orange_liberties = min(orange_group_libs) if orange_group_libs else 8
    
    # 5. 영토 수 계산
    blue_territory, orange_territory = calculate_territory(board)
    
    # 6. 승리 원인 분류 자동 태깅
    loser = ORANGE if winner == BLUE else BLUE
    winner_territory = blue_territory if winner == BLUE else orange_territory
    loser_territory = orange_territory if winner == BLUE else blue_territory
    loser_liberties = orange_liberties if winner == BLUE else blue_liberties
    
    tag = None
    if termination == "CAPTURE":
        if winner_territory < loser_territory:
            tag = "COMEBACK_CAPTURE"
        elif loser_liberties <= 2:
            tag = "DOMINANT_CAPTURE"
        elif 3 <= loser_liberties <= 5:
            tag = "NARROW_CAPTURE"
        else:
            tag = "TERRITORY_PLUS_CAPTURE"
            
    game_id = os.path.basename(game_file)
    
    return {
        "game_id": game_id,
        "winner": winner,
        "win_reason": win_reason,
        "termination": termination,
        "final_move": final_move,
        "blue_stones": blue_stones,
        "orange_stones": orange_stones,
        "blue_liberties": blue_liberties,
        "orange_liberties": orange_liberties,
        "blue_territory": blue_territory,
        "orange_territory": orange_territory,
        "winner_territory": winner_territory,
        "loser_territory": loser_territory,
        "loser_liberties": loser_liberties,
        "tag": tag
    }

def run_analysis_pipeline(game_dir="data/games"):
    os.makedirs("analysis", exist_ok=True)
    game_files = sorted(glob.glob(os.path.join(game_dir, "game_*.json")))
    
    if not game_files:
        print(f"No game files found in '{game_dir}' to analyze.")
        return
        
    analyses = []
    for f in game_files:
        try:
            res = analyze_single_game(f)
            analyses.append(res)
        except Exception as e:
            print(f"Failed to analyze game {f}: {e}")
            
    num_games = len(analyses)
    if num_games == 0:
        return
        
    # 통계 계산
    tag_counts = {
        "DOMINANT_CAPTURE": 0,
        "NARROW_CAPTURE": 0,
        "COMEBACK_CAPTURE": 0,
        "TERRITORY_PLUS_CAPTURE": 0
    }
    
    total_blue_territory = 0
    total_orange_territory = 0
    total_blue_liberties = 0
    total_orange_liberties = 0
    
    for a in analyses:
        if a["tag"] in tag_counts:
            tag_counts[a["tag"]] += 1
            
        total_blue_territory += a["blue_territory"]
        total_orange_territory += a["orange_territory"]
        total_blue_liberties += a["blue_liberties"]
        total_orange_liberties += a["orange_liberties"]
        
    avg_blue_territory = total_blue_territory / num_games
    avg_orange_territory = total_orange_territory / num_games
    avg_blue_liberties = total_blue_liberties / num_games
    avg_orange_liberties = total_orange_liberties / num_games
    
    territory_diff = avg_blue_territory - avg_orange_territory
    liberty_diff = avg_blue_liberties - avg_orange_liberties
    
    # 텍스트 출력용 차이 포맷팅
    terr_diff_str = f"BLUE {territory_diff:+.1f}" if territory_diff >= 0 else f"ORANGE {abs(territory_diff):+.1f}"
    lib_diff_str = f"BLUE {liberty_diff:+.1f}" if liberty_diff >= 0 else f"ORANGE {abs(liberty_diff):+.1f}"
    
    # 1. JSON 리포트 생성
    report_data = {
        "summary": {
            "total_games": num_games,
            "tag_counts": tag_counts,
            "tag_percentages": {k: f"{v/num_games*100:.1f}%" for k, v in tag_counts.items()},
            "average_stats": {
                "blue_territory": round(avg_blue_territory, 2),
                "orange_territory": round(avg_orange_territory, 2),
                "blue_liberties": round(avg_blue_liberties, 2),
                "orange_liberties": round(avg_orange_liberties, 2),
                "territory_difference": terr_diff_str,
                "liberty_difference": lib_diff_str
            }
        },
        "details": analyses
    }
    
    with open("analysis/final_position_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    # 2. Markdown 리포트 생성
    md_content = []
    md_content.append("# Great Kingdom AI - Final Position Analysis Report\n")
    md_content.append("## 📊 Executive Summary\n")
    md_content.append(f"* **Analyzed Games**: {num_games} games")
    md_content.append(f"* **Average Territory**: BLUE {avg_blue_territory:.1f} vs ORANGE {avg_orange_territory:.1f}")
    md_content.append(f"* **Average Liberties**: BLUE {avg_blue_liberties:.1f} vs ORANGE {avg_orange_liberties:.1f}\n")
    
    md_content.append("## 🏆 Winning Tag Classification\n")
    md_content.append("| Tag Name | Game Count | Percentage | Description |")
    md_content.append("| :--- | :---: | :---: | :--- |")
    md_content.append(f"| **DOMINANT_CAPTURE** | {tag_counts['DOMINANT_CAPTURE']} | {tag_counts['DOMINANT_CAPTURE']/num_games*100:.1f}% | Opponent liberties <= 2 at termination |")
    md_content.append(f"| **NARROW_CAPTURE** | {tag_counts['NARROW_CAPTURE']} | {tag_counts['NARROW_CAPTURE']/num_games*100:.1f}% | Opponent liberties 3~5 at termination |")
    md_content.append(f"| **COMEBACK_CAPTURE** | {tag_counts['COMEBACK_CAPTURE']} | {tag_counts['COMEBACK_CAPTURE']/num_games*100:.1f}% | Winner territory < Loser territory at capture |")
    md_content.append(f"| **TERRITORY_PLUS_CAPTURE** | {tag_counts['TERRITORY_PLUS_CAPTURE']} | {tag_counts['TERRITORY_PLUS_CAPTURE']/num_games*100:.1f}% | Winner territory >= Loser territory at capture |")
    md_content.append("\n")
    
    md_content.append("## 📉 Difference Analysis\n")
    md_content.append(f"* **Average Territory Difference**: `{terr_diff_str}`")
    md_content.append(f"* **Average Liberty Difference**: `{lib_diff_str}`\n")
    
    md_content.append("## 🔍 Detailed Positions\n")
    md_content.append("| Game ID | Winner | Termination | Moves | BLUE Stones | ORANGE Stones | BLUE Libs | ORANGE Libs | BLUE Terr | ORANGE Terr | Tag |")
    md_content.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for a in analyses:
        tag_str = f"`{a['tag']}`" if a["tag"] else "`-`"
        winner_str = "BLUE" if a["winner"] == 1 else "ORANGE"
        md_content.append(
            f"| {a['game_id']} | {winner_str} | {a['termination']} | {a['final_move']} | "
            f"{a['blue_stones']} | {a['orange_stones']} | {a['blue_liberties']} | {a['orange_liberties']} | "
            f"{a['blue_territory']} | {a['orange_territory']} | {tag_str} |"
        )
        
    with open("analysis/final_position_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    # 콘솔에 지시서 포맷과 거의 유사하게 출력
    print("\n===== FINAL POSITION ANALYSIS =====")
    for k, v in tag_counts.items():
        print(f"{k}: {v} games ({v/num_games*100:.1f}%)")
    print()
    print("Average Territory Difference:")
    print(terr_diff_str)
    print()
    print("Average Liberty Difference:")
    print(lib_diff_str)
    print("===================================\n")

if __name__ == "__main__":
    run_analysis_pipeline()
