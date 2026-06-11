import os
import json
import sys
import glob

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from engine.territory import calculate_territory

def replay_to_move(moves, target_count):
    """지정된 수순까지만 재생한 GameState를 반환"""
    game = GameState()
    game.is_copy = True
    
    for idx in range(min(target_count, len(moves))):
        move = moves[idx]
        if game.game_over:
            break
        if move == "pass":
            game.play_pass()
        else:
            game.play_move(move[0], move[1])
            
    return game

def analyze_territory_trend(game_file):
    with open(game_file, "r", encoding="utf-8") as f:
        game_data = json.load(f)
        
    winner = game_data["winner"]
    win_reason = game_data.get("win_reason", "CAPTURE")
    moves = game_data["moves"]
    N = len(moves)
    
    # 마지막 시점들(N-5부터 N-1번째 수 직후 상태)
    # 0-indexed moves에서:
    # N-5번째 착수 완료 상태 = moves[:N-4]
    # N-4번째 착수 완료 상태 = moves[:N-3]
    # N-3번째 착수 완료 상태 = moves[:N-2]
    # N-2번째 착수 완료 상태 = moves[:N-1]
    # N-1번째 착수 완료 상태 = moves[:-1] (마지막 결정타 직전 상태)
    
    trend = []
    # 최소 5수 미만인 짧은 대국의 경우 가능한 수만큼만 추적
    start_offset = max(0, N - 5)
    
    for i in range(start_offset, N):
        # i번째 수순까지 둔 보드 상태 복원
        # 단, 마지막 수 N-1번째는 결정타 직전 상태(즉 moves[:-1])를 의미함
        target_len = i if i == N - 1 else i + 1
        
        game = replay_to_move(moves, target_len)
        blue_terr, orange_terr = calculate_territory(game.board)
        
        # 1-indexed 수순 번호
        move_num = i if i == N - 1 else i + 1
        is_capture_point = (i == N - 1)
        
        winner_terr = blue_terr if winner == BLUE else orange_terr
        loser_terr = orange_terr if winner == BLUE else blue_terr
        
        trend.append({
            "move": move_num,
            "is_capture_point": is_capture_point,
            "blue_territory": blue_terr,
            "orange_territory": orange_terr,
            "winner_territory": winner_terr,
            "loser_territory": loser_terr,
            "diff": blue_terr - orange_terr
        })
        
    return {
        "game_id": os.path.basename(game_file),
        "winner": winner,
        "win_reason": win_reason,
        "total_moves": N,
        "trend": trend
    }

def run_trend_analysis(game_dir="data/games"):
    os.makedirs("analysis", exist_ok=True)
    game_files = sorted(glob.glob(os.path.join(game_dir, "game_*.json")))
    
    if not game_files:
        print(f"No game files found in '{game_dir}' to analyze.")
        return
        
    game_trends = []
    for f in game_files:
        try:
            res = analyze_territory_trend(f)
            game_trends.append(res)
        except Exception as e:
            print(f"Failed to analyze territory trend for {f}: {e}")
            
    num_games = len(game_trends)
    if num_games == 0:
        return
        
    # 지표 집계 변수들
    total_points = 0
    winner_dominant_points = 0
    loser_dominant_at_capture_count = 0
    total_capture_games = 0
    
    sum_blue_orange_diff = 0
    sum_winner_loser_diff = 0
    
    detailed_stats = []
    
    for gt in game_trends:
        trend = gt["trend"]
        if not trend:
            continue
            
        winner = gt["winner"]
        win_reason = gt["win_reason"]
        
        # 각 시점별 누적 계산
        for pt in trend:
            total_points += 1
            if pt["winner_territory"] > pt["loser_territory"]:
                winner_dominant_points += 1
            sum_blue_orange_diff += pt["diff"]
            sum_winner_loser_diff += (pt["winner_territory"] - pt["loser_territory"])
            
        # 캡처 직전 시점 분석 (마지막 수 직전 상태 = trend[-1])
        if win_reason == "CAPTURE" and trend:
            total_capture_games += 1
            final_pt = trend[-1]
            if final_pt["loser_territory"] > final_pt["winner_territory"]:
                loser_dominant_at_capture_count += 1
                
    # 1. 캡처 직전 평균 영토 차이
    avg_diff_blue_orange = sum_blue_orange_diff / total_points if total_points > 0 else 0
    avg_diff_winner_loser = sum_winner_loser_diff / total_points if total_points > 0 else 0
    
    avg_diff_str = f"BLUE {avg_diff_blue_orange:+.2f}" if avg_diff_blue_orange >= 0 else f"ORANGE {abs(avg_diff_blue_orange):+.2f}"
    avg_diff_wl_str = f"WINNER {avg_diff_winner_loser:+.2f}" if avg_diff_winner_loser >= 0 else f"LOSER {abs(avg_diff_winner_loser):+.2f}"
    
    # 2. 승리자 영토 우세 비율 (5수 시점 전체 모수 기준)
    winner_dominant_ratio = (winner_dominant_points / total_points * 100.0) if total_points > 0 else 0.0
    
    # 3. 패배자 영토 우세 상태에서 캡처당한 비율 (CAPTURE로 끝난 대국 기준)
    loser_dominant_capture_ratio = (loser_dominant_at_capture_count / total_capture_games * 100.0) if total_capture_games > 0 else 0.0
    
    # 4. 영토 우세였지만 역전패한 사례 수 (CAPTURE로 끝난 대국 기준)
    comeback_capture_count = loser_dominant_at_capture_count
    
    # JSON 파일 생성
    report_json = {
        "summary": {
            "total_games": num_games,
            "total_capture_games": total_capture_games,
            "average_territory_difference_blue_orange": avg_diff_str,
            "average_territory_difference_winner_loser": avg_diff_wl_str,
            "winner_territory_dominant_percentage": f"{winner_dominant_ratio:.1f}%",
            "loser_dominant_at_capture_percentage": f"{loser_dominant_capture_ratio:.1f}%",
            "comeback_capture_count": comeback_capture_count
        },
        "details": game_trends
    }
    
    with open("analysis/territory_trend_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
        
    # MD 파일 생성
    md_content = []
    md_content.append("# Great Kingdom AI - Territory Trend Analysis Report\n")
    md_content.append("## 📊 Summary Metrics\n")
    md_content.append(f"1. **캡처 직전 평균 영토 차이**: `{avg_diff_str}` (승리자 기준: `{avg_diff_wl_str}`)")
    md_content.append(f"2. **승리자 영토 우세 비율**: `{winner_dominant_ratio:.1f}%` (마지막 5수 동안의 시점 기준)")
    md_content.append(f"3. **패배자 영토 우세 상태에서 캡처당한 비율**: `{loser_dominant_capture_ratio:.1f}%` (총 {total_capture_games}판 중 {comeback_capture_count}판)")
    md_content.append(f"4. **영토 우세였지만 역전패한 사례 수**: `{comeback_capture_count}`판\n")
    
    # AI 성향 분석 추가 판별
    # 영토 크기가 전반적으로 0에 가깝다면 "순수 캡처 지향", 영토 점수가 유의미하게 형성되면 "영토 전략 수행 중"으로 판별
    total_territory_accum = 0
    for gt in game_trends:
        for pt in gt["trend"]:
            total_territory_accum += (pt["blue_territory"] + pt["orange_territory"])
            
    avg_game_territory = total_territory_accum / total_points if total_points > 0 else 0
    
    md_content.append("## 🧠 AI Strategy Classification\n")
    if avg_game_territory < 1.0:
        md_content.append("> [!WARNING]")
        md_content.append("> **순수 캡처 게임 수행 중 (Pure Capture Play)**")
        md_content.append(f"> * 대국 종반부 평균 영토 점수가 `{avg_game_territory:.2f}`점으로 극히 낮습니다. AI가 영토 전략을 거의 의식하지 않고 즉각적인 상대방의 활로 포위 및 방어에만 몰두하고 있는 상태입니다.")
    else:
        md_content.append("> [!NOTE]")
        md_content.append("> **영토 전략 및 캡처 병행 수행 중 (Territory + Capture Play)**")
        md_content.append(f"> * 대국 종반부 평균 영토 점수가 `{avg_game_territory:.2f}`점으로 유의미하게 형성되었습니다. AI가 영토 거점을 확보하려는 전술과 상대 돌을 포획하려는 전술을 균형 있게 병행 중입니다.")
    md_content.append("\n")
    
    # 상세 대국별 영토 추이 마크다운 테이블 추가
    md_content.append("## 🔍 Detailed Game Trends (Last 5 Moves)\n")
    for gt in game_trends:
        md_content.append(f"### 🎮 {gt['game_id']} (Total Moves: {gt['total_moves']}, Winner: {'BLUE' if gt['winner'] == BLUE else 'ORANGE'})\n")
        md_content.append("| Move Num | BLUE Territory | ORANGE Territory | Winner Territory | Loser Territory | Difference | Status |")
        md_content.append("| :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for pt in gt["trend"]:
            status_str = "**CAPTURE POINT (Pre-capture)**" if pt["is_capture_point"] else "Ongoing"
            md_content.append(
                f"| Move {pt['move']} | {pt['blue_territory']} | {pt['orange_territory']} | "
                f"{pt['winner_territory']} | {pt['loser_territory']} | {pt['diff']:+d} | {status_str} |"
            )
        md_content.append("\n")
        
    with open("analysis/territory_trend_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    # 콘솔 출력
    print("\n===== TERRITORY TREND ANALYSIS =====")
    print(f"1. 캡처 직전 평균 영토 차이:\n   {avg_diff_str}")
    print()
    print(f"2. 승리자 영토 우세 비율:\n   {winner_dominant_ratio:.1f}%")
    print()
    print(f"3. 패배자 영토 우세 상태에서 캡처당한 비율:\n   {loser_dominant_capture_ratio:.1f}%")
    print()
    print(f"4. 영토 우세였지만 역전패한 사례 수:\n   {comeback_capture_count} games")
    print()
    print("AI Strategy Classification:")
    if avg_game_territory < 1.0:
        print("   PURE CAPTURE PLAY (Average territory score is near zero)")
    else:
        print("   TERRITORY + CAPTURE HYBRID PLAY")
    print("=====================================\n")

if __name__ == "__main__":
    run_trend_analysis()
