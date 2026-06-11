import streamlit as st
import os
import json
import time

from engine.game_state import GameState
from engine.board import BLUE, ORANGE, EMPTY
from ai.minimax import find_best_move
from engine.territory import calculate_territory

# 페이지 설정
st.set_page_config(
    page_title="Great Kingdom AI - Web Test Room",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "game" not in st.session_state:
    st.session_state.game = GameState()
    st.session_state.game.is_copy = True
if "game_moves" not in st.session_state:
    st.session_state.game_moves = []  # 순수 좌표 및 패스 히스토리 [ [r,c], "pass", ... ]
if "logs" not in st.session_state:
    st.session_state.logs = []
if "show_territory" not in st.session_state:
    st.session_state.show_territory = False
if "test_records" not in st.session_state:
    record_path = "analysis/human_test_records.json"
    if os.path.exists(record_path):
        try:
            with open(record_path, "r", encoding="utf-8") as f:
                st.session_state.test_records = json.load(f)
        except:
            st.session_state.test_records = []
    else:
        st.session_state.test_records = []

def reset_game():
    st.session_state.game = GameState()
    st.session_state.game.is_copy = True
    st.session_state.game_moves = []
    st.session_state.logs = ["대국이 새롭게 기동되었습니다. 당신은 BLUE(선공)입니다."]
    # 쿼리 파라미터 초기화
    st.query_params.clear()
    st.rerun()

def save_record(game_num, winner_str, reason, total_moves, memo):
    new_record = {
        "game_num": game_num,
        "winner": winner_str,
        "reason": reason,
        "total_moves": total_moves,
        "memo": memo,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.test_records.append(new_record)
    
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/human_test_records.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.test_records, f, indent=2, ensure_ascii=False)
        
    md_content = ["# Great Kingdom AI - Human Test Suite Records\n"]
    md_content.append("| 대국 번호 | 승자 | 종료 원인 | 총 수순 | Human 테스트 메모 | 일시 |")
    md_content.append("| :---: | :---: | :---: | :---: | :--- | :--- |")
    for r in st.session_state.test_records:
        md_content.append(
            f"| Game {r['game_num']} | {r['winner']} | {r['reason']} | {r['total_moves']} | {r['memo']} | {r['timestamp']} |"
        )
    with open("analysis/human_test_records.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    st.success(f"Game {game_num} 결과가 성공적으로 기록되었습니다!")

# ----------------- URL 쿼리 파라미터 착수 감지 -----------------
# 양방향 통신 트릭: 클릭 시 iframe에서 부모 주소를 '?click=r_c'로 리다이렉트함
click_param = st.query_params.get("click")
if click_param and not st.session_state.game.game_over and st.session_state.game.current_player == BLUE:
    try:
        r_str, c_str = click_param.split("_")
        r, c = int(r_str), int(c_str)
        
        # 착수 유효성 검사 (이미 돌이 놓인 곳 제외)
        if st.session_state.game.board.get(r, c) == EMPTY:
            captured_occurred = st.session_state.game.play_move(r, c)
            st.session_state.game_moves.append([r, c])
            
            log_msg = f"BLUE (Human) -> ({r}, {c})"
            if captured_occurred:
                log_msg += " [CAPTURE 발생!]"
            st.session_state.logs.append(log_msg)
            
            # 파라미터 지우고 리렌더링
            st.query_params.clear()
            st.rerun()
    except Exception as e:
        st.error(f"착수 처리 오류: {e}")

# ----------------- UI 레이아웃 -----------------
st.title("🏯 Great Kingdom AI - Human Test Room")

# 사이드바 설정
st.sidebar.header("⚙️ 컨트롤 타워")
if st.sidebar.button("🔄 게임 리셋 (새 대국 시작)", use_container_width=True):
    reset_game()

# 현재 대국 상태 및 영토 획득
game = st.session_state.game
board = game.board
blue_territory, orange_territory = calculate_territory(board)

# 영토 시각화 체크박스 (ON/OFF)
st.session_state.show_territory = st.sidebar.toggle(
    "🗺️ 영토 분석 모드 시각화 (ON/OFF)", 
    value=st.session_state.show_territory
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 현재 상태 계기판")
current_player_str = "🔵 BLUE (Human)" if game.current_player == BLUE else "🟠 ORANGE (AI)"
st.sidebar.write(f"**현재 수순**: {len(st.session_state.game_moves) + 1} 수째")
st.sidebar.write(f"**현재 차례**: {current_player_str}")
st.sidebar.write(f"🔵 **Human (BLUE) 영토**: {blue_territory} 칸")
st.sidebar.write(f"🟠 **AI (ORANGE) 영토**: {orange_territory} 칸")
st.sidebar.write(f"Pass 연속 횟수: {game.consecutive_passes}회")

# Human Pass 버튼
if not game.game_over and game.current_player == BLUE:
    if st.sidebar.button("🏳️ Pass (한 차례 넘기기)", use_container_width=True):
        game.play_pass()
        st.session_state.game_moves.append("pass")
        st.session_state.logs.append("BLUE (Human) -> PASS")
        st.query_params.clear()
        st.rerun()

# 최근 로그 내역
st.sidebar.markdown("---")
st.sidebar.subheader("📜 수순 로그")
log_text = "\n".join(st.session_state.logs[-15:])
st.sidebar.text_area("최근 수순 내역", log_text, height=180, disabled=True)

# AI 백그라운드 연산 즉시 기동
if not game.game_over and game.current_player == ORANGE:
    with st.spinner("🤖 AI가 미니맥스(Depth 3) 수읽기 중..."):
        # AI 생각 연산 시간을 일부러 주고 현실감을 높임
        time.sleep(0.5)
        ai_move = find_best_move(game, depth=3)
        
        if ai_move == "pass":
            game.play_pass()
            st.session_state.game_moves.append("pass")
            st.session_state.logs.append("ORANGE (AI) -> PASS")
        else:
            captured_occurred = game.play_move(ai_move[0], ai_move[1])
            st.session_state.game_moves.append([ai_move[0], ai_move[1]])
            
            log_msg = f"ORANGE (AI) -> ({ai_move[0]}, {ai_move[1]})"
            if captured_occurred:
                log_msg += " [CAPTURE 발생!]"
            st.session_state.logs.append(log_msg)
            
        st.query_params.clear()
        st.rerun()

# ----------------- HTML5/CSS3 바둑판 렌더링 -----------------
# 9x9 격자판 데이터 구성
board_list = []
for r in range(9):
    row_data = []
    for c in range(9):
        cell_val = board.get(r, c)
        is_neutral = (r, c) in [(2,4), (4,2), (4,6), (6,4), (4,4)]
        
        # 영토 여부 판단
        # calculate_territory는 전체 맵의 영토 배정을 하지 않고 총합만 리턴할 수 있으므로,
        # 각 빈 칸에 대해 영토 판정을 하는 BFS 로직을 내장하거나, 
        # engine/territory.py의 상세 맵 구조를 직접 참조하여 칠합니다.
        row_data.append({
            "r": r,
            "c": c,
            "val": cell_val,
            "neutral": is_neutral
        })
    board_list.append(row_data)

# 영토 맵 구체화 (시각화 켜져 있을 때 각 격자의 영토 귀속 파악)
territory_map = {}
if st.session_state.show_territory:
    # engine.territory 내부의 상세 계산 로직을 본따서 각 빈칸의 영토 상태를 구함
    # calculation_territory의 상세 BFS를 임시로 구현해 각 칸을 칠합니다.
    from engine.territory import calculate_territory
    # 빈칸들을 그룹화하여 영토 파악
    empty_cells = []
    for r in range(9):
        for c in range(9):
            if board.get(r, c) == EMPTY and (r, c) not in [(2,4), (4,2), (4,6), (6,4), (4,4)]:
                empty_cells.append((r, c))
                
    # 간단한 BFS로 인접 돌 체크
    visited = set()
    for start in empty_cells:
        if start in visited:
            continue
        group = []
        queue = [start]
        visited.add(start)
        borders = set()
        
        while queue:
            curr_r, curr_c = queue.pop(0)
            group.append((curr_r, curr_c))
            
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    val = board.get(nr, nc)
                    if val == EMPTY:
                        if (nr, nc) not in visited and (nr, nc) not in [(2,4), (4,2), (4,6), (6,4), (4,4)]:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                    elif val in (BLUE, ORANGE):
                        borders.add(val)
                        
        # 영토 판정
        if len(borders) == 1:
            owner = list(borders)[0]
            for gr, gc in group:
                territory_map[f"{gr}_{gc}"] = owner

# 마지막 수 좌표 파악
last_move_coord = None
if st.session_state.game_moves:
    last = st.session_state.game_moves[-1]
    if last != "pass":
        last_move_coord = last  # [r, c]

# HTML 및 CSS 조립
# 바둑판은 목조 베이지 무늬(#f0cb85)와 짙은 선(#4b3b28)으로 디자인
# 좌표축(숫자 0~8)을 포함한 10x10 CSS Grid 구성
grid_html = """
<div class="board-container">
    <div class="board-grid">
        <!-- 빈칸 헤더 (좌측 상단 모서리) -->
        <div class="coord-label header-label"></div>
        
        <!-- 열 좌표 축 -->
        <div class="coord-label header-label">0</div>
        <div class="coord-label header-label">1</div>
        <div class="coord-label header-label">2</div>
        <div class="coord-label header-label">3</div>
        <div class="coord-label header-label">4</div>
        <div class="coord-label header-label">5</div>
        <div class="coord-label header-label">6</div>
        <div class="coord-label header-label">7</div>
        <div class="coord-label header-label">8</div>
"""

for r in range(9):
    # 행 좌표 축 (좌측)
    grid_html += f'<div class="coord-label row-label">{r}</div>'
    for c in range(9):
        cell_info = board_list[r][c]
        val = cell_info["val"]
        is_neutral = cell_info["neutral"]
        
        # 클래스 설정
        classes = ["board-cell"]
        
        # 영토 시각화 배경칠하기
        terr_owner = territory_map.get(f"{r}_{c}")
        if terr_owner == BLUE:
            classes.append("bg-blue-territory")
        elif terr_owner == ORANGE:
            classes.append("bg-orange-territory")
            
        class_str = " ".join(classes)
        
        # 돌 렌더링
        stone_html = ""
        is_last = last_move_coord and last_move_coord[0] == r and last_move_coord[1] == c
        last_move_glow = " last-move-glow" if is_last else ""
        
        if val == BLUE:
            stone_html = f'<div class="stone stone-blue{last_move_glow}"></div>'
        elif val == ORANGE:
            stone_html = f'<div class="stone stone-orange{last_move_glow}"></div>'
        elif is_neutral:
            # 검은 성 아이콘 또는 검은 돌 표시
            stone_html = f'<div class="stone stone-neutral{last_move_glow}">🏰</div>'
        elif is_last:
            # 빈 곳인데 마지막 수인 경우 (그럴 일은 거의 없지만 안전 조치)
            stone_html = f'<div class="last-move-marker"></div>'
            
        # 클릭 이벤트 연동
        # 버튼처럼 작동하되 일체감 있는 보드로 렌더링
        is_clickable = (val == EMPTY and not game.game_over and game.current_player == BLUE)
        clickable_attr = f'onclick="cellClicked({r}, {c})"' if is_clickable else ""
        cursor_style = " cursor-pointer" if is_clickable else " cursor-not-allowed"
        
        grid_html += f"""
        <div class="{class_str}{cursor_style}" {clickable_attr}>
            {stone_html}
        </div>
        """

grid_html += """
    </div>
</div>
"""

# 스타일 정의
board_css = """
<style>
.board-container {
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #f7f9fa;
    padding: 20px;
    border-radius: 12px;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.05);
}
.board-grid {
    display: grid;
    grid-template-columns: 30px repeat(9, 50px);
    grid-template-rows: 30px repeat(9, 50px);
    gap: 0px;
    background-color: #ebc178;
    border: 3px solid #3d2d1d;
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}
.coord-label {
    display: flex;
    justify-content: center;
    align-items: center;
    font-weight: bold;
    font-size: 14px;
    color: #3d2d1d;
    user-select: none;
}
.header-label {
    border-bottom: 2px solid transparent;
}
.row-label {
    border-right: 2px solid transparent;
}
.board-cell {
    position: relative;
    border: 1px solid #4a3b2c;
    background-color: #f2d096;
    display: flex;
    justify-content: center;
    align-items: center;
    box-sizing: border-box;
    transition: background-color 0.2s;
}
/* 영토 배경 */
.bg-blue-territory {
    background-color: rgba(59, 118, 225, 0.35) !important;
}
.bg-orange-territory {
    background-color: rgba(233, 87, 63, 0.35) !important;
}
/* 커서 활성화 */
.cursor-pointer {
    cursor: pointer;
}
.cursor-pointer:hover {
    background-color: #e5bd7c;
}
.cursor-not-allowed {
    cursor: not-allowed;
}
/* 돌 디자인 */
.stone {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 20px;
    user-select: none;
    z-index: 5;
    box-shadow: 2px 3px 6px rgba(0,0,0,0.3);
}
.stone-blue {
    background: radial-gradient(circle at 30% 30%, #5d9cec, #3b76e1);
    border: 1px solid #2e5cb8;
}
.stone-orange {
    background: radial-gradient(circle at 30% 30%, #fc6e51, #e9573f);
    border: 1px solid #c83d27;
}
.stone-neutral {
    background: radial-gradient(circle at 30% 30%, #4f5d75, #2d3748);
    border: 1px solid #1a202c;
    color: white;
}
/* 마지막 수 강조 */
.last-move-glow {
    border: 3px solid #ffce54 !important;
    box-shadow: 0 0 15px #ffce54, 2px 3px 6px rgba(0,0,0,0.3) !important;
    animation: pulse 1.0s infinite alternate;
}
.last-move-marker {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: #ffce54;
    box-shadow: 0 0 8px #ffce54;
}
@keyframes pulse {
    0% { transform: scale(1.0); }
    100% { transform: scale(1.05); }
}
</style>

<script>
function cellClicked(r, c) {
    // 부모 Streamlit 창의 URL을 변경하여 클릭 정보를 파라미터로 송신
    window.parent.location.href = window.parent.location.origin + window.parent.location.pathname + "?click=" + r + "_" + c;
}
</script>
"""

# 메인 레이아웃 렌더링
col_game_board, col_panel = st.columns([1.6, 1])

with col_game_board:
    # 격자 보드 HTML 출력
    st.components.v1.html(board_css + grid_html, height=580, scrolling=False)

# ----------------- 종료 및 기록 폼 렌더링 -----------------
if game.game_over:
    st.balloons()
    winner_str = "🔵 BLUE (Human)" if game.winner == BLUE else "🟠 ORANGE (AI)"
    reason_str = "TERRITORY (영토 판정 승)" if game.consecutive_passes >= 2 else "CAPTURE (돌 포획 승)"
    
    st.info(f"### 🏆 대국 종료!\n* **승리자**: {winner_str}\n* **종료 원인**: {reason_str}\n* **총 수순**: {len(st.session_state.game_moves)} 수")
    
    with col_panel:
        st.subheader("📝 Human Test Suite 기록 양식")
        st.write("실전 대국에서 관측하신 AI의 평가함수 결함이나 전술 맹점을 기록해 주세요.")
        
        with st.form("record_form"):
            game_num = st.number_input("대국 차례 (Game 1~10)", min_value=1, max_value=20, value=len(st.session_state.test_records) + 1)
            winner_input = st.selectbox("실제 대국 승자", ["BLUE (Human)", "ORANGE (AI)"])
            reason_input = st.selectbox("종료 사유 분류", ["CAPTURE", "TERRITORY"])
            total_moves_input = st.number_input("최종 수순 수", min_value=1, max_value=150, value=len(st.session_state.game_moves))
            memo_input = st.text_area("AI 약점 피드백 메모", placeholder="예: AI가 중앙 돌에만 집착하여 가장자리 방어에 실패함 / 단수 위협을 감지 못함")
            
            submitted = st.form_submit_button("💾 대국 테스트 결과 저장")
            if submitted:
                save_record(game_num, winner_input, reason_input, total_moves_input, memo_input)

# ----------------- 누적 테스트 기록 조회 -----------------
with col_panel:
    st.subheader("📋 Human Test Suite 누적 목록")
    if st.session_state.test_records:
        for r in st.session_state.test_records:
            st.write(f"**Game {r['game_num']}** | 승자: `{r['winner']}` | 원인: `{r['reason']}` | 수순: `{r['total_moves']}수`")
            st.write(f"> *메모: {r['memo']}*")
            st.markdown("---")
    else:
        st.write("아직 등록된 테스트 기록이 없습니다. 대국을 완료하고 기록을 저장해 주세요.")
