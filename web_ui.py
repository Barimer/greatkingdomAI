import streamlit as st
import os
import json
import time

from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from ai.minimax import find_best_move
from engine.territory import calculate_territory
from engine.capture import get_group, get_liberties

# 페이지 설정
st.set_page_config(
    page_title="Great Kingdom AI - Human Test Mode",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "game" not in st.session_state:
    st.session_state.game = GameState()
    st.session_state.game.is_copy = True  # 콘솔에 캡처로그 등 불필요한 출력 방지
if "logs" not in st.session_state:
    st.session_state.logs = []
if "test_records" not in st.session_state:
    # 기존 기록 파일 로드 시도
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
    st.session_state.logs = ["게임이 리셋되었습니다. 당신은 BLUE(선공)입니다."]
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
    
    # 1. JSON 저장
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/human_test_records.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.test_records, f, indent=2, ensure_ascii=False)
        
    # 2. Markdown 저장
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
st.title("🎮 Great Kingdom AI - Human Test Mode (Web UI)")

# 사이드바 설정
st.sidebar.header("⚙️ 컨트롤 타워")
if st.sidebar.button("🔄 게임 리셋 (새 대국 시작)", use_container_width=True):
    reset_game()

# 현재 게임 상태 분석
game = st.session_state.game
board = game.board

# 영토 계산
blue_territory, orange_territory = calculate_territory(board)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 현재 대국 상태")
current_player_str = "🔵 BLUE (Human)" if game.current_player == BLUE else "🟠 ORANGE (AI)"
st.sidebar.write(f"**현재 턴**: {len(st.session_state.logs) + 1} 수째")
st.sidebar.write(f"**현재 차례**: {current_player_str}")
st.sidebar.write(f"🔵 **Human (BLUE) 영토**: {blue_territory} 칸")
st.sidebar.write(f"🟠 **AI (ORANGE) 영토**: {orange_territory} 칸")
st.sidebar.write(f"Pass 연속 횟수: {game.consecutive_passes}회")

# 패스(Pass) 기능
if not game.game_over and game.current_player == BLUE:
    if st.sidebar.button("🏳️ Pass (한 차례 넘기기)", use_container_width=True):
        game.play_pass()
        st.session_state.logs.append("BLUE (Human) -> PASS")
        st.rerun()

# 로그 표시 영역
st.sidebar.markdown("---")
st.sidebar.subheader("📜 대국 로그")
log_text = "\n".join(st.session_state.logs[-15:])  # 최근 15개 로그만 사이드바에 표시
st.sidebar.text_area("최근 수순 내역", log_text, height=200, disabled=True)


# Main 화면 분할
col_board, col_panel = st.columns([2, 1])

with col_board:
    st.subheader("🏁 9x9 그레이트 킹덤 바둑판")
    
    # 커스텀 보드 렌더링
    # 9x9 그리드를 Streamlit columns로 구성
    for r in range(9):
        cols = st.columns(9)
        for c in range(9):
            cell = board.get(r, c)
            
            # 중립 성 여부 판단
            # 룰엔진의 중립성 좌표는 (2,4), (4,2), (4,6), (6,4), (4,4) 입니다.
            is_neutral = (r, c) in [(2,4), (4,2), (4,6), (6,4), (4,4)]
            
            # 표시할 유니코드 기호 결정
            if cell == BLUE:
                symbol = "🔵"
            elif cell == ORANGE:
                symbol = "🟠"
            elif is_neutral:
                symbol = "⚫"
            else:
                symbol = "⚪"
                
            # 버튼 클릭 이벤트 처리
            # 유저의 돌은 BLUE이며, game_over가 아니며, BLUE 차례일 때만 클릭 활성화
            btn_key = f"cell_{r}_{c}"
            is_disabled = game.game_over or game.current_player != BLUE or cell in (BLUE, ORANGE)
            
            if cols[c].button(symbol, key=btn_key, help=f"({r}, {c}) 좌표", disabled=is_disabled, use_container_width=True):
                # 유저 착수 진행
                captured_occurred = game.play_move(r, c)
                log_msg = f"BLUE (Human) -> ({r}, {c})"
                if captured_occurred:
                    log_msg += " [CAPTURE 발생]"
                st.session_state.logs.append(log_msg)
                st.rerun()

# AI 연산 구동부
# Human의 착수가 끝나 현재 턴이 AI(ORANGE)이고 게임오버가 아닐 때 백그라운드 연산 즉시 기동
if not game.game_over and game.current_player == ORANGE:
    with col_panel:
        with st.spinner("🤖 AI가 미니맥스(Depth 3) 수읽기 중..."):
            ai_move = find_best_move(game, depth=3)
            
            if ai_move == "pass":
                game.play_pass()
                st.session_state.logs.append("ORANGE (AI) -> PASS")
            else:
                captured_occurred = game.play_move(ai_move[0], ai_move[1])
                log_msg = f"ORANGE (AI) -> ({ai_move[0]}, {ai_move[1]})"
                if captured_occurred:
                    log_msg += " [CAPTURE 발생]"
                st.session_state.logs.append(log_msg)
                
            st.rerun()

# ----------------- 종료 화면 렌더링 -----------------
if game.game_over:
    st.balloons()
    winner_str = "🔵 BLUE (Human)" if game.winner == BLUE else "🟠 ORANGE (AI)"
    
    # 종료 사유 도출
    if game.consecutive_passes >= 2:
        reason_str = "TERRITORY (영토 판정 승리)"
    else:
        reason_str = "CAPTURE (돌 포획 서든데스 승리)"
        
    st.info(f"### 🏆 대국 종료!\n* **승리자**: {winner_str}\n* **종료 원인**: {reason_str}\n* **총 수순**: {len(st.session_state.logs)} 수")
    
    with col_panel:
        st.subheader("📝 Human Test Suite 기록 양식")
        st.write("실전 대국에서 AI의 성능 맹점이나 특이 행동을 관찰하여 기록해 주세요.")
        
        # 기록 폼 구성
        with st.form("record_form"):
            game_num = st.number_input("대국 차례 (예: 1~10)", min_value=1, max_value=20, value=len(st.session_state.test_records) + 1)
            winner_input = st.selectbox("실제 대국 승자", ["BLUE (Human)", "ORANGE (AI)"])
            reason_input = st.selectbox("종료 사유 분류", ["CAPTURE", "TERRITORY"])
            total_moves_input = st.number_input("최종 수순 수", min_value=1, max_value=150, value=len(st.session_state.logs))
            memo_input = st.text_area("AI 약점 및 전술 분석 피드백 메모", placeholder="예: AI가 중앙 돌에만 집착하여 가장자리 방어에 실패함 / 단수 위협을 무시함 등")
            
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
