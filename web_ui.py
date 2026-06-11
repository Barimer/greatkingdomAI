import streamlit as st
import os
import json
import time

from engine.game_state import GameState
from engine.board import BLUE, ORANGE, EMPTY, NEUTRAL
from ai.minimax import find_best_move, LAST_AI_DECISION
from engine.territory import calculate_territory

# 페이지 설정
st.set_page_config(
    page_title="Great Kingdom AI - Web Test Room",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 디버그 콘솔 출력 여부 플래그
DEBUG = False

# 세션 상태 초기화
if "game" not in st.session_state:
    st.session_state.game = GameState()
    st.session_state.game.is_copy = False
if "game_moves" not in st.session_state:
    st.session_state.game_moves = []
if "logs" not in st.session_state:
    st.session_state.logs = ["대국이 새롭게 기동되었습니다. 당신은 BLUE(선공)입니다. 교차점을 클릭하여 착수하세요."]
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
    st.session_state.game.is_copy = False
    st.session_state.game_moves = []
    st.session_state.logs = ["대국이 새롭게 기동되었습니다. 당신은 BLUE(선공)입니다. 교차점을 클릭하여 착수하세요."]
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

# ----------------- SVG 바둑판 생성기 (오직 시각 효과 집중) -----------------
def generate_board_svg(game, territory_map, last_move_coord):
    board = game.board
    size = 500
    grid_start = 40
    grid_gap = 50
    
    svg = []
    # 둥근 모서리와 그림자가 들어간 고풍스러운 바둑판 나무 배경 및 검은색 외곽선
    svg.append(f'<svg width="500" height="500" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" style="background-color: #df9e51; border-radius: 12px; box-shadow: 0 12px 24px rgba(0,0,0,0.45); border: 5px solid #2e1e0a; display: block;">')
    
    # 3D 효과를 위한 그라데이션 및 필터 정의
    svg.append('<defs>')
    # 3D BLUE 돌
    svg.append('<radialGradient id="blueStone" cx="35%" cy="35%" r="65%">')
    svg.append('<stop offset="0%" stop-color="#99c2ff" />')
    svg.append('<stop offset="30%" stop-color="#3385ff" />')
    svg.append('<stop offset="85%" stop-color="#003d99" />')
    svg.append('<stop offset="100%" stop-color="#001a4d" />')
    svg.append('</radialGradient>')
    # 3D ORANGE 돌
    svg.append('<radialGradient id="orangeStone" cx="35%" cy="35%" r="65%">')
    svg.append('<stop offset="0%" stop-color="#ffc299" />')
    svg.append('<stop offset="30%" stop-color="#ff8533" />')
    svg.append('<stop offset="85%" stop-color="#b34700" />')
    svg.append('<stop offset="100%" stop-color="#4d1f00" />')
    svg.append('</radialGradient>')
    # 마지막 수 노란색 글로우 필터
    svg.append('<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">')
    svg.append('<feGaussianBlur stdDeviation="5" result="blur" />')
    svg.append('<feComposite in="SourceGraphic" in2="blur" operator="over" />')
    svg.append('</filter>')
    # 돌 하단 입체 그림자 필터
    svg.append('<filter id="shadow" x="-15%" y="-15%" width="130%" height="130%">')
    svg.append('<feDropShadow dx="3" dy="5" stdDeviation="3.5" flood-opacity="0.5" />')
    svg.append('</filter>')
    svg.append('</defs>')
    
    # 격자선 그리기 (9x9)
    for r in range(9):
        y = grid_start + r * grid_gap
        svg.append(f'<line x1="{grid_start}" y1="{y}" x2="{grid_start + 8*grid_gap}" y2="{y}" stroke="#2e1e0a" stroke-width="1.8" />')
    for c in range(9):
        x = grid_start + c * grid_gap
        svg.append(f'<line x1="{x}" y1="{grid_start}" x2="{x}" y2="{grid_start + 8*grid_gap}" stroke="#2e1e0a" stroke-width="1.8" />')
        
    # 화점 (Star Points) (2,2), (2,6), (4,4), (6,2), (6,6)
    star_points = [(2,2), (2,6), (4,4), (6,2), (6,6)]
    for r_s, c_s in star_points:
        x_s = grid_start + c_s * grid_gap
        y_s = grid_start + r_s * grid_gap
        svg.append(f'<circle cx="{x_s}" cy="{y_s}" r="5" fill="#2e1e0a" />')
        
    # 좌표 문자 출력 (상단 0~8, 좌측 0~8)
    for i in range(9):
        # 상단 열 번호
        x_t = grid_start + i * grid_gap
        svg.append(f'<text x="{x_t}" y="20" font-size="13" font-weight="black" fill="#2e1e0a" text-anchor="middle" dominant-baseline="central" style="user-select:none; font-family:\'Outfit\',\'Inter\',sans-serif;">{i}</text>')
        # 좌측 행 번호
        y_t = grid_start + i * grid_gap
        svg.append(f'<text x="20" y="{y_t}" font-size="13" font-weight="black" fill="#2e1e0a" text-anchor="middle" dominant-baseline="central" style="user-select:none; font-family:\'Outfit\',\'Inter\',sans-serif;">{i}</text>')
        
    # 영토 분석 모드 활성화 시 은은한 배경 사각형 렌더링
    if st.session_state.show_territory:
        for r in range(9):
            for c in range(9):
                key = f"{r}_{c}"
                if key in territory_map:
                    owner = territory_map[key]
                    x_t = grid_start + c * grid_gap
                    y_t = grid_start + r * grid_gap
                    color = "#2b7fff" if owner == BLUE else "#ff732b"
                    svg.append(f'<rect x="{x_t - 22}" y="{y_t - 22}" width="44" height="44" rx="8" fill="{color}" fill-opacity="0.32" stroke="{color}" stroke-width="1.5" stroke-dasharray="3,3" />')
                    
    # 돌 렌더링 (BLUE / ORANGE / NEUTRAL)
    for r in range(9):
        for c in range(9):
            cell_val = board.get(r, c)
            x = grid_start + c * grid_gap
            y = grid_start + r * grid_gap
            
            is_last = (last_move_coord and last_move_coord[0] == r and last_move_coord[1] == c)
            
            # 마지막 수 주위 노란색 글로우 테두리 강조
            if is_last and cell_val in (BLUE, ORANGE):
                svg.append(f'<circle cx="{x}" cy="{y}" r="22" stroke="#ffe600" stroke-width="5" fill="none" filter="url(#glow)" />')
                
            if cell_val == BLUE:
                svg.append(f'<circle cx="{x}" cy="{y}" r="19" fill="url(#blueStone)" filter="url(#shadow)" />')
            elif cell_val == ORANGE:
                svg.append(f'<circle cx="{x}" cy="{y}" r="19" fill="url(#orangeStone)" filter="url(#shadow)" />')
            elif cell_val == NEUTRAL:
                # 중립 성 3D 엠블럼처럼 렌더링
                svg.append(f'<text x="{x}" y="{y}" font-size="30" text-anchor="middle" dominant-baseline="central" filter="url(#shadow)" style="user-select:none;">🏰</text>')
                
    svg.append('</svg>')
    return "\n".join(svg)

# ----------------- UI 레이아웃 -----------------
st.title("🏰 Great Kingdom AI - Human Test Room")

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
        st.rerun()

# 최근 로그 내역
st.sidebar.markdown("---")
st.sidebar.subheader("📜 수순 로그")
log_text = "\n".join(st.session_state.logs[-15:])
st.sidebar.text_area("최근 수순 내역", log_text, height=180, disabled=True)

# AI 사고 로그 패널 표시
if LAST_AI_DECISION["move"] is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI 사고 분석 로그")
    st.sidebar.write(f"**AI Move**: `{LAST_AI_DECISION['move']}`")
    st.sidebar.write(f"**Evaluation**: `{LAST_AI_DECISION['score']:+.2f}`")
    st.sidebar.write(f"**Depth**: `{LAST_AI_DECISION['depth']}`")

# AI 백그라운드 연산 즉시 기동
if not game.game_over and game.current_player == ORANGE:
    with st.spinner("🤖 AI가 미니맥스(Depth 3) 수읽기 중..."):
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
            
        st.rerun()

# ----------------- 바둑판 전용 절대배치 CSS -----------------
board_style = """
<style>
/* 바둑판 고정 크기 500px 래퍼 */
.board-wrapper {
    position: relative !important;
    width: 500px !important;
    height: 500px !important;
    margin: 0 auto !important;
}

/* 배경 SVG 레이어 */
.board-background-svg {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 500px !important;
    height: 500px !important;
    z-index: 1 !important;
    pointer-events: none !important;
}

/* 위에 얹어지는 투명 버튼 오버레이 */
.button-grid-overlay {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 500px !important;
    height: 500px !important;
    z-index: 2 !important;
    display: flex !important;
    flex-direction: column !important;
}

/* 행(stHorizontalBlock) 레이아웃 설정 */
.button-grid-overlay div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 0px !important;
    width: 500px !important;
    margin: 0px !important;
    padding: 0px !important;
}

/* 첫 번째 좌표 행 높이 */
.button-grid-overlay div[data-testid="stHorizontalBlock"]:first-child {
    height: 40px !important;
}
/* 나머지 실제 격자 9개 행 높이 */
.button-grid-overlay div[data-testid="stHorizontalBlock"]:not(:first-child) {
    height: 50px !important;
}

/* 열(column) 레이아웃 설정 */
.button-grid-overlay div[data-testid="column"] {
    padding: 0px !important;
    margin: 0px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

/* 첫 번째 좌표 열 너비 */
.button-grid-overlay div[data-testid="column"]:first-child {
    width: 40px !important;
    min-width: 40px !important;
    max-width: 40px !important;
}
/* 나머지 실제 격자 9개 열 너비 */
.button-grid-overlay div[data-testid="column"]:not(:first-child) {
    width: 50px !important;
    min-width: 50px !important;
    max-width: 50px !important;
}

/* 투명 클릭 버튼 디자인 */
.button-grid-overlay .stButton > button {
    width: 44px !important;
    height: 44px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 50% !important;
    padding: 0px !important;
    margin: 0px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: background-color 0.12s, border 0.12s;
}

/* 빈 교차점 호버 시 파란 돌 가이드 잔상 표시 */
.button-grid-overlay .stButton > button:not(:disabled):hover {
    background-color: rgba(0, 85, 255, 0.28) !important;
    border: 2.5px solid #0055ff !important;
}

/* 이미 돌이 있는 곳이나 상대 턴일 때 비활성화 투명 상태 유지 */
.button-grid-overlay .stButton > button:disabled {
    background: transparent !important;
    border: none !important;
    cursor: default !important;
}
</style>
"""
st.markdown(board_style, unsafe_allow_html=True)

# ----------------- 영토 분석 맵 계산 -----------------
territory_map = {}
if st.session_state.show_territory:
    empty_cells = []
    for r in range(9):
        for c in range(9):
            if board.get(r, c) == EMPTY and (r, c) not in [(2,4), (4,2), (4,6), (6,4), (4,4)]:
                empty_cells.append((r, c))
                
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
                        
        if len(borders) == 1:
            owner = list(borders)[0]
            for gr, gc in group:
                territory_map[f"{gr}_{gc}"] = owner

# 마지막 수 좌표 파악
last_move_coord = None
if st.session_state.game_moves:
    last = st.session_state.game_moves[-1]
    if last != "pass":
        last_move_coord = last

# ----------------- 바둑판 렌더링 -----------------
col_game_board, col_panel = st.columns([1.5, 1])

with col_game_board:
    st.subheader("🏁 9x9 그레이트 킹덤 바둑판")
    
    # 바둑판 래퍼 열기 (상대 배치 컨테이너)
    st.markdown('<div class="board-wrapper">', unsafe_allow_html=True)
    
    # 1. 고해상도 SVG 바둑판을 배경 레이어로 배치
    board_svg = generate_board_svg(game, territory_map, last_move_coord)
    st.markdown(f'<div class="board-background-svg">{board_svg}</div>', unsafe_allow_html=True)
    
    # 2. 투명 버튼 오버레이 레이어 겹쳐서 배치
    st.markdown('<div class="button-grid-overlay">', unsafe_allow_html=True)
    
    # 상단 열 좌표를 위한 빈 줄(투명 높이 확보용)
    cols_header = st.columns(10)
    for c in range(10):
        cols_header[c].write("")
        
    # 9x9 격자 투명 버튼들 배치
    for r in range(9):
        cols = st.columns(10)
        # 첫 열 빈 공간 (좌측 좌표축 영역 공간 확보)
        cols[0].write("")
        
        for c in range(9):
            cell_val = board.get(r, c)
            is_disabled = game.game_over or game.current_player != BLUE or cell_val in (BLUE, ORANGE, NEUTRAL)
            
            # 투명 버튼 클릭 시 즉시 착수 및 새로고침
            if cols[c+1].button("", key=f"cell_{r}_{c}", disabled=is_disabled):
                captured_occurred = game.play_move(r, c)
                st.session_state.game_moves.append([r, c])
                
                log_msg = f"BLUE (Human) -> ({r}, {c})"
                if captured_occurred:
                    log_msg += " [CAPTURE 발생!]"
                st.session_state.logs.append(log_msg)
                st.rerun()
                
    st.markdown('</div>', unsafe_allow_html=True) # button-grid-overlay 닫기
    st.markdown('</div>', unsafe_allow_html=True) # board-wrapper 닫기

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
