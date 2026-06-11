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
    if DEBUG:
        print("\n[INITIAL STATE] GameState created:")
        print(st.session_state.game.board.grid)
        print()
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
    st.session_state.game.is_copy = False
    if DEBUG:
        print("\n[RESET STATE] GameState created:")
        print(st.session_state.game.board.grid)
        print()
    st.session_state.game_moves = []
    st.session_state.logs = ["대국이 새롭게 기동되었습니다. 당신은 BLUE(선공)입니다."]
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
        st.rerun()

# 최근 로그 내역
st.sidebar.markdown("---")
st.sidebar.subheader("📜 수순 로그")
log_text = "\n".join(st.session_state.logs[-15:])
st.sidebar.text_area("최근 수순 내역", log_text, height=180, disabled=True)

# AI 사고 로그 패널 표시 (콘솔 출력 금지 대응)
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

# ----------------- 바둑판 오버라이드 CSS -----------------
# Streamlit의 컬럼 및 버튼 마진을 0으로 뭉개서 완벽한 격자판을 렌더링
board_style = """
<style>
/* 바둑판 전용 st.columns 간 격자 여백 제거 */
div[data-testid="column"] div[data-testid="stHorizontalBlock"] {
    gap: 0px !important;
    max-width: 480px !important; /* 바둑판 전체 너비를 최대 480px로 제한 */
    margin: 0 auto !important;   /* 가운데 정렬 */
}

/* 바둑판 전용 각 열(column) 패딩/마진 제거 및 정렬 */
div[data-testid="column"] div[data-testid="column"] {
    padding: 0px !important;
    margin: 0px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

/* 바둑판 전용 버튼 디자인 (반응형 100% 너비 및 1:1 비율) */
div[data-testid="column"] div[data-testid="column"] .stButton > button {
    width: 100% !important;
    aspect-ratio: 1 / 1 !important;
    padding: 0px !important;
    margin: 0px !important;
    border-radius: 0px !important;
    border: 1px solid #4a3b2c !important;
    background-color: #f2d096 !important;
    font-size: 22px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    box-sizing: border-box !important;
    transition: background-color 0.1s;
}

div[data-testid="column"] div[data-testid="column"] .stButton > button:hover {
    background-color: #e5bd7c !important;
    border-color: #4a3b2c !important;
}

/* 좌표 라벨 텍스트 스타일 */
.coord-label {
    font-size: 16px;
    font-weight: bold;
    color: #4a3b2c;
    user-select: none;
    text-align: center;
    width: 100% !important;
    aspect-ratio: 1 / 1 !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    box-sizing: border-box !important;
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

# UI 렌더링 직전 보드 셀 정보 콘솔 출력 (DEBUG 모드일 때만 수행)
if DEBUG:
    print("\n--- UI RENDER BOARD INSPECTION ---")
    for r_inspect in range(9):
        for c_inspect in range(9):
            print(r_inspect, c_inspect, board.grid[r_inspect][c_inspect])
    print("-----------------------------------\n")

# ----------------- 바둑판 렌더링 (Streamlit native) -----------------
col_game_board, col_panel = st.columns([1.6, 1])

with col_game_board:
    st.subheader("🏁 9x9 그레이트 킹덤 바둑판")
    
    # 1. 상단 열 좌표 인덱스 출력 (0~8)
    cols_header = st.columns(10)
    cols_header[0].markdown('<div class="coord-label"></div>', unsafe_allow_html=True) # 좌측 상단 모서리 빈칸
    for c in range(9):
        cols_header[c+1].markdown(f'<div class="coord-label">{c}</div>', unsafe_allow_html=True)
        
    # 2. 바둑판 격자 렌더링 (0~8 행)
    for r in range(9):
        cols = st.columns(10)
        # 좌측 행 좌표 인덱스 출력
        cols[0].markdown(f'<div class="coord-label">{r}</div>', unsafe_allow_html=True)
        
        for c in range(9):
            cell_val = board.get(r, c)
            is_neutral = (cell_val == NEUTRAL)
            is_last = last_move_coord and last_move_coord[0] == r and last_move_coord[1] == c
            
            # 유니코드 입체 기호 결정
            if cell_val == BLUE:
                symbol = "🔵"
                if is_last:
                    symbol = "🔵🎯" # 마지막 착수 강조
            elif cell_val == ORANGE:
                symbol = "orange" # 아래에서 특수 아이콘 대체
                symbol = "🟠"
                if is_last:
                    symbol = "🟠🎯" # 마지막 착수 강조
            elif is_neutral:
                symbol = "🏰" # 중립 성 표시
            else:
                # 빈칸일 때 영토 시각화 맵과 마지막 수 여부에 따라 마크 결정
                terr_owner = territory_map.get(f"{r}_{c}")
                if terr_owner == BLUE:
                    symbol = "🔹" # 연파랑 영토
                elif terr_owner == ORANGE:
                    symbol = "🔸" # 연주황 영토
                elif is_last:
                    symbol = "🎯"
                else:
                    symbol = "➕" # 바둑판 교차점 느낌
            
            # 버튼 클릭 액션 핸들링 (착수는 EMPTY 일 때만 활성화)
            is_disabled = game.game_over or game.current_player != BLUE or cell_val in (BLUE, ORANGE, NEUTRAL)
            
            if cols[c+1].button(symbol, key=f"cell_{r}_{c}", disabled=is_disabled):
                # 유저 착수 처리
                captured_occurred = game.play_move(r, c)
                st.session_state.game_moves.append([r, c])
                
                log_msg = f"BLUE (Human) -> ({r}, {c})"
                if captured_occurred:
                    log_msg += " [CAPTURE 발생!]"
                st.session_state.logs.append(log_msg)
                st.rerun()

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
