import streamlit as st
import os
import json
import time
import base64
import random

from engine.game_state import GameState
from engine.board import BLUE, ORANGE, EMPTY, NEUTRAL
from ai.minimax import find_best_move, LAST_AI_DECISION
from engine.territory import calculate_territory
from streamlit_image_coordinates import streamlit_image_coordinates

# 페이지 설정
st.set_page_config(
    page_title="Great Kingdom AI - Web Test Room",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 반응형 레이아웃용 CSS 주입 (창 크기가 1200px 미만일 때 세로 배치 전환)
st.markdown(
    """
    <style>
    @media (max-width: 1200px) {
        div.stHorizontalBlock, [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            flex-wrap: wrap !important;
            display: flex !important;
        }
        div.stColumn, [data-testid="column"], [data-testid="stColumn"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
            flex-basis: 100% !important;
        }
    }
    /* Rerun trigger 버튼 숨김 */
    div.stButton > button[key="rerun_trigger_btn"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Rerun Trigger 버튼 정의 (JS에서 클릭 트리거용)
if st.button("Rerun Trigger", key="rerun_trigger_btn"):
    st.rerun()

# JS로 브라우저 너비를 강제 동기화 (새로고침 없는 rerun 활용)
import streamlit.components.v1 as components
components.html(
    """
    <script>
        function syncWidth() {
            const currentWidth = window.parent.innerWidth;
            const queryParams = new URLSearchParams(window.parent.location.search);
            if (queryParams.get('win_width') !== String(currentWidth)) {
                queryParams.set('win_width', currentWidth);
                const newurl = window.parent.location.protocol + "//" + window.parent.location.host + window.parent.location.pathname + '?' + queryParams.toString();
                window.parent.history.pushState({path:newurl}, '', newurl);
                
                // Rerun Trigger 버튼 클릭
                const buttons = window.parent.document.querySelectorAll('button');
                for (let i = 0; i < buttons.length; i++) {
                    if (buttons[i].textContent.includes('Rerun Trigger')) {
                        buttons[i].click();
                        break;
                    }
                }
            }
        }
        setTimeout(syncWidth, 300);
        window.parent.addEventListener('resize', syncWidth);
    </script>
    """,
    height=0
)

# 디버그 콘솔 출력 여부 플래그
DEBUG = False

def log_state(stage):
    try:
        import time
        game = st.session_state.get("game")
        player = game.current_player if game else "None"
        game_over = game.game_over if game else "None"
        moves_len = len(st.session_state.get("game_moves", []))
        game_id = id(game) if game else 0
        last_click = st.session_state.get("last_processed_click")
        preview = st.session_state.get("preview_coord")
        
        log_line = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STAGE: {stage:<15} | "
            f"PLAYER: {player} | GAMEOVER: {game_over} | MOVES: {moves_len:<2} | "
            f"GAME_ID: {game_id} | LAST_CLICK: {last_click} | PREVIEW: {preview}\n"
        )
        import os
        os.makedirs("analysis", exist_ok=True)
        with open("analysis/debug_log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        pass

# 세션 상태 초기화
if "game" not in st.session_state:
    st.session_state.game = GameState()
    st.session_state.game.is_copy = False
if "game_moves" not in st.session_state:
    st.session_state.game_moves = []
if "logs" not in st.session_state:
    st.session_state.logs = ["대국이 새롭게 기동되었습니다. 당신은 BLUE(선공)입니다. 바둑판의 교차점을 클릭하여 착수하세요."]
if "show_territory" not in st.session_state:
    st.session_state.show_territory = False
if "last_processed_click" not in st.session_state:
    st.session_state.last_processed_click = None
if "preview_coord" not in st.session_state:
    st.session_state.preview_coord = None
if "last_board_image" not in st.session_state:
    st.session_state.last_board_image = None
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

log_state("A_PAGE_LOAD")

def reset_game():
    st.session_state.game = GameState()
    st.session_state.game.is_copy = False
    st.session_state.game_moves = []
    st.session_state.logs = ["대국이 새롭게 기동되었습니다. 당신은 BLUE(선공)입니다. 바둑판의 교차점을 클릭하여 착수하세요."]
    st.session_state.last_processed_click = None
    st.session_state.preview_coord = None
    st.session_state.last_board_image = None
    st.rerun()

def save_record(game_num, winner_str, reason, total_moves, memo):
    new_record = {
        "game_num": game_num,
        "winner": winner_str,
        "reason": reason,
        "total_moves": total_moves,
        "moves": st.session_state.game_moves.copy(),
        "memo": memo,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.test_records.append(new_record)
    
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/human_test_records.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.test_records, f, indent=2, ensure_ascii=False)
        
    md_content = ["# Great Kingdom AI - Human Test Suite Records\n"]
    md_content.append("| 대국 번호 | 승자 | 종료 원인 | 총 수순 | 상세 행마 | Human 테스트 메모 | 일시 |")
    md_content.append("| :---: | :---: | :---: | :---: | :--- | :--- | :--- |")
    for r in st.session_state.test_records:
        moves_str = str(r.get("moves", []))
        md_content.append(
            f"| Game {r['game_num']} | {r['winner']} | {r['reason']} | {r['total_moves']} | `{moves_str}` | {r['memo']} | {r['timestamp']} |"
        )
    with open("analysis/human_test_records.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    st.success(f"Game {game_num} 결과가 성공적으로 기록되었습니다!")

# ----------------- PNG 바둑판 생성기 (PIL) -----------------
def generate_board_png(game, territory_map, last_move_coord, preview_coord=None, show_territory=False):
    from PIL import Image, ImageDraw, ImageFont
    
    # 700x700 RGBA 이미지 생성 (배경 나무색 #df9e51)
    img = Image.new("RGBA", (700, 700), (223, 158, 81, 255))
    draw = ImageDraw.Draw(img)
    
    grid_start = 70
    grid_gap = 70
    
    # 폰트 로드
    font_path = "C:\\Windows\\Fonts\\malgun.ttf"  # 맑은 고딕
    if not os.path.exists(font_path):
        font_path = "C:\\Windows\\Fonts\\arial.ttf"
        
    try:
        font_coord = ImageFont.truetype(font_path, 22)
        font_neutral = ImageFont.truetype(font_path, 26)
    except:
        font_coord = ImageFont.load_default()
        font_neutral = ImageFont.load_default()

    # 격자선 그리기 (9x9)
    # 외곽선은 두께 4px, 내부선은 2px
    for r in range(9):
        y = grid_start + r * grid_gap
        width = 4 if (r == 0 or r == 8) else 2
        draw.line([(grid_start, y), (grid_start + 8 * grid_gap, y)], fill=(46, 30, 10, 255), width=width)
        
    for c in range(9):
        x = grid_start + c * grid_gap
        width = 4 if (c == 0 or c == 8) else 2
        draw.line([(x, grid_start), (x, grid_start + 8 * grid_gap)], fill=(46, 30, 10, 255), width=width)
        
    # 화점 그리기
    star_points = [(2,2), (2,6), (4,4), (6,2), (6,6)]
    for r_s, c_s in star_points:
        cx = grid_start + c_s * grid_gap
        cy = grid_start + r_s * grid_gap
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(46, 30, 10, 255))
        
    # 좌표 그리기 (0~8)
    for i in range(9):
        coord_str = str(i)
        # 상단 좌표
        tx = grid_start + i * grid_gap
        ty = 35
        try:
            draw.text((tx, ty), coord_str, fill=(46, 30, 10, 255), font=font_coord, anchor="mm")
        except:
            draw.text((tx - 6, ty - 12), coord_str, fill=(46, 30, 10, 255), font=font_coord)
            
        # 좌측 좌표
        lx = 35
        ly = grid_start + i * grid_gap
        try:
            draw.text((lx, ly), coord_str, fill=(46, 30, 10, 255), font=font_coord, anchor="mm")
        except:
            draw.text((lx - 6, ly - 12), coord_str, fill=(46, 30, 10, 255), font=font_coord)

    # 영토 표시 시각화
    if show_territory:
        for r in range(9):
            for c in range(9):
                key = f"{r}_{c}"
                if key in territory_map:
                    owner = territory_map[key]
                    x_t = grid_start + c * grid_gap
                    y_t = grid_start + r * grid_gap
                    # 은은한 반투명 색상 박스 (BLUE: 파랑, ORANGE: 주황)
                    color = (43, 127, 255, 80) if owner == BLUE else (255, 115, 43, 80)
                    draw.rectangle([x_t - 32, y_t - 32, x_t + 32, y_t + 32], fill=color, outline=color[:3] + (160,), width=2)

    # 3D 구체(돌) 그리기 헬퍼 함수
    def draw_3d_stone(cx, cy, color_type, alpha=255):
        r_max = 28
        offset_x, offset_y = -7, -7
        
        for i in range(r_max, 0, -1):
            ratio = i / r_max
            if color_type == BLUE:
                red = int(0 + (100 - 0) * (1 - ratio))
                green = int(50 + (180 - 50) * (1 - ratio))
                blue = int(180 + (255 - 180) * (1 - ratio))
            elif color_type == ORANGE:
                red = int(160 + (255 - 160) * (1 - ratio))
                green = int(60 + (170 - 60) * (1 - ratio))
                blue = int(0 + (100 - 0) * (1 - ratio))
            else: # NEUTRAL
                red = int(20 + (80 - 20) * (1 - ratio))
                green = int(20 + (80 - 20) * (1 - ratio))
                blue = int(20 + (80 - 20) * (1 - ratio))
                
            curr_cx = cx + int(offset_x * (1 - ratio))
            curr_cy = cy + int(offset_y * (1 - ratio))
            draw.ellipse([curr_cx - i, curr_cy - i, curr_cx + i, curr_cy + i], fill=(red, green, blue, alpha))

    # 돌 렌더링 (BLUE / ORANGE / NEUTRAL)
    board = game.board
    for r in range(9):
        for c in range(9):
            cell_val = board.get(r, c)
            cx = grid_start + c * grid_gap
            cy = grid_start + r * grid_gap
            
            is_last = (last_move_coord and last_move_coord[0] == r and last_move_coord[1] == c)
            
            if cell_val == BLUE:
                draw_3d_stone(cx, cy, BLUE)
            elif cell_val == ORANGE:
                draw_3d_stone(cx, cy, ORANGE)
            elif cell_val == NEUTRAL:
                draw_3d_stone(cx, cy, NEUTRAL)
                try:
                    draw.text((cx, cy), "N", fill=(255, 255, 255, 255), font=font_neutral, anchor="mm")
                except:
                    draw.text((cx - 8, cy - 15), "N", fill=(255, 255, 255, 255), font=font_neutral)
            
            # 마지막 수 노란 테두리 강조
            if is_last and cell_val in (BLUE, ORANGE):
                draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], outline=(255, 230, 0, 255), width=4)

    # Hover(미리보기) 위치 표시 (반투명 BLUE 돌)
    if preview_coord is not None:
        pr, pc = preview_coord
        if board.get(pr, pc) == EMPTY:
            cx = grid_start + pc * grid_gap
            cy = grid_start + pr * grid_gap
            draw_3d_stone(cx, cy, BLUE, alpha=120)
            draw.ellipse([cx - 29, cy - 29, cx + 29, cy + 29], outline=(255, 255, 255, 180), width=2)
            
    return img.convert("RGB")

# 현재 대국 상태 및 영토 획득
game = st.session_state.game
board = game.board
blue_territory, orange_territory = calculate_territory(board)

# AI 연산 중인지 체크
is_ai_turn = not game.game_over and game.current_player == ORANGE

# 사이드바 설정
st.sidebar.header("⚙️ 컨트롤 타워")
if st.sidebar.button("🔄 게임 리셋 (새 대국 시작)", use_container_width=True, disabled=is_ai_turn):
    reset_game()

# 영토 시각화 체크박스 (ON/OFF)
st.session_state.show_territory = st.sidebar.toggle(
    "🗺️ 영토 분석 모드 시각화 (ON/OFF)", 
    value=st.session_state.show_territory,
    disabled=is_ai_turn
)

if "prev_show_territory" not in st.session_state:
    st.session_state.prev_show_territory = st.session_state.show_territory
elif st.session_state.show_territory != st.session_state.prev_show_territory:
    st.session_state.prev_show_territory = st.session_state.show_territory
    log_state("E_TOGGLE_CLICK")

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
# st.query_params에서 가로 너비 획득 (반응형 board_size 계산)
win_width_str = st.query_params.get("win_width", "1200")
try:
    win_width = int(win_width_str)
except:
    win_width = 1200

# 세로 배치 모드(1200px 이하)인지 가로 배치 모드(1200px 초과)인지에 따라 가용 폭 계산 다각화
if win_width <= 1200:
    # 세로 배치일 때는 대국 현황판의 가로 폭 간섭이 없으므로 본문 전체 가로 폭을 확보 (사이드바 약 330px 마진 제외)
    available_width = win_width - 330
else:
    # 가로 배치일 때는 [ 바둑판 ] [ 대국 현황판 ] 으로 분할하므로, 
    # 대국 현황판 가로 최소 폭(약 450px)과 사이드바(약 330px)를 동시 감안
    available_width = win_width - 780

# 바둑판 크기는 min(700, available_width)이되, 조작 편의를 위해 최소 350px 보장
board_size = max(350, min(700, available_width))

col_game_board, col_panel = st.columns([1.5, 1])

with col_game_board:
    st.subheader("🏁 9x9 그레이트 킹덤 바둑판")
    
    # 700x700 PNG 이미지 생성 (캐싱 폴백 적용)
    try:
        board_img = generate_board_png(
            game, 
            territory_map, 
            last_move_coord, 
            preview_coord=st.session_state.preview_coord, 
            show_territory=st.session_state.show_territory
        )
        st.session_state.last_board_image = board_img
    except Exception as e:
        if st.session_state.last_board_image is not None:
            board_img = st.session_state.last_board_image
        else:
            from PIL import Image
            board_img = Image.new("RGB", (700, 700), (223, 158, 81, 255))
    
    # PIL Image 객체를 직접 전달 (반응형 board_size 적용)
    value = streamlit_image_coordinates(
        board_img,
        width=board_size,
        height=board_size,
        key="board_coordinates"
    )
    
    # 클릭 좌표 처리 및 2단계 착수 프리뷰 연산 (AI 연산 중일 때는 차단)
    if value is not None and not is_ai_turn:
        click_key = (value["x"], value["y"])
        
        if click_key != st.session_state.last_processed_click:
            st.session_state.last_processed_click = click_key
            
            x, y = value["x"], value["y"]
            
            # 반응형 board_size 스케일링 비율에 따른 클릭 좌표 물리 픽셀(700px 기준) 복원
            scale = board_size / 700.0
            orig_x = x / scale
            orig_y = y / scale
            
            # 클릭 좌표 ➔ 격자 좌표 변환 공식 (오차 반올림, 간격 70px)
            c_click = round((orig_x - 70) / 70)
            r_click = round((orig_y - 70) / 70)
            
            center_x = 70 + c_click * 70
            center_y = 70 + r_click * 70
            dist = (((orig_x - center_x) ** 2 + (orig_y - center_y) ** 2) ** 0.5) * scale
            
            # 클릭이 격자 유효 범위 안에 있는 경우 (스케일 반영 임계값 28px)
            if dist <= 28 * scale and 0 <= r_click < 9 and 0 <= c_click < 9:
                if not game.game_over and game.current_player == BLUE:
                    # 합법 수 여부 검사 (상대 영토 착수 금지 룰 준수)
                    from ai.minimax import get_legal_moves
                    if (r_click, c_click) in get_legal_moves(game):
                        # 2단계 착수 확인 체크
                        if st.session_state.preview_coord == (r_click, c_click):
                            # 실제 착수 진행
                            captured_occurred = game.play_move(r_click, c_click)
                            log_state("B_HUMAN_PLAY")
                            st.session_state.game_moves.append([r_click, c_click])
                            
                            log_msg = f"BLUE (Human) -> ({r_click}, {c_click})"
                            if captured_occurred:
                                log_msg += " [CAPTURE 발생!]"
                            st.session_state.logs.append(log_msg)
                            st.session_state.preview_coord = None
                            st.rerun()
                        else:
                            # 1단계 미리보기 상태 설정
                            st.session_state.preview_coord = (r_click, c_click)
                            st.rerun()
                    else:
                        st.session_state.preview_coord = None
                        st.rerun()
            else:
                st.session_state.preview_coord = None
                st.rerun()
                
    # AI 연산 진행률 및 착수 시간 표시를 위한 placeholder (바둑판 밑에 배치)
    ai_status_placeholder = st.empty()
    ai_progress_bar_placeholder = st.empty()
 
# ----------------- 현재 상태 패널 -----------------
with col_panel:
    st.markdown("### 📊 대국 현황판")
    
    current_player_str = "🔵 BLUE (Human)" if game.current_player == BLUE else "🟠 ORANGE (AI)"
    
    # 2단계 착수 가이드 문구
    if not game.game_over and game.current_player == BLUE:
        if st.session_state.preview_coord:
            st.warning(f"👉 선택한 좌표 **{st.session_state.preview_coord}**를 바둑판에서 한 번 더 클릭하면 착수가 확정됩니다.")
        else:
            st.info("💡 바둑판의 교차점을 클릭하여 착수할 위치를 미리보기 하세요.")
            
    m1, m2 = st.columns(2)
    m1.metric("현재 차례", current_player_str)
    m2.metric("진행 수순", f"{len(st.session_state.game_moves)} 수째")
    
    m3, m4 = st.columns(2)
    m3.metric("🔵 BLUE 영토", f"{blue_territory} 칸")
    m4.metric("🟠 ORANGE 영토", f"{orange_territory} 칸")
    
    ai_score_str = f"{LAST_AI_DECISION['score']:+.2f}" if LAST_AI_DECISION["score"] is not None else "N/A"
    last_move_str = str(last_move_coord) if last_move_coord else "없음"
    
    m5, m6 = st.columns(2)
    m5.metric("🤖 AI 평가값", ai_score_str)
    m6.metric("🎯 마지막 착수 좌표", last_move_str)
    
    # 최근 10수 로그만 깔끔하게 출력
    st.markdown("---")
    st.markdown("#### 📜 최근 10수 로그")
    recent_logs = st.session_state.logs[-10:]
    for log_item in recent_logs:
        st.write(f"- {log_item}")
        
    # Pass 버튼
    if not game.game_over and game.current_player == BLUE:
        st.markdown("---")
        if st.button("🏳️ Pass (한 차례 넘기기)", use_container_width=True):
            game.play_pass()
            st.session_state.game_moves.append("pass")
            st.session_state.logs.append("BLUE (Human) -> PASS")
            st.session_state.preview_coord = None
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

# AI 백그라운드 연산 즉시 기동 (모든 UI가 다 그려진 후 최하단에서 동작)
if not game.game_over and game.current_player == ORANGE:
    # 바둑판 하단에 미리 배치된 placeholder 재사용
    status_placeholder = ai_status_placeholder
    progress_bar = ai_progress_bar_placeholder
    
    with status_placeholder.container():
        st.markdown("##### 🤖 AI 수읽기 중 (Depth 3)...")
        info_text = st.empty()
        
    from ai.minimax import get_legal_moves, copy_game_state, alphabeta, LAST_AI_DECISION
    log_state("C_AI_BEFORE")
    
    legal_moves = get_legal_moves(game)
    move_scores = []
    alpha = -float("inf")
    beta = float("inf")
    target_player = ORANGE
    depth = 3
    
    start_time = time.time()
    total_moves = len(legal_moves)
    
    for idx, move in enumerate(legal_moves):
        # 경과 및 남은 시간 계산
        elapsed = time.time() - start_time
        if idx > 0:
            avg_time = elapsed / idx
            eta = avg_time * (total_moves - idx)
            info_text.caption(
                f"**진행률**: {idx}/{total_moves} ({int(idx/total_moves*100)}%) | "
                f"**경과 시간**: {elapsed:.1f}초 | "
                f"**예상 남은 시간**: {eta:.1f}초"
            )
        else:
            info_text.caption(
                f"**진행률**: {idx}/{total_moves} (0%) | "
                f"**경과 시간**: {elapsed:.1f}초 | "
                f"**예상 남은 시간**: 계산 중..."
            )
            
        progress_bar.progress(float(idx / total_moves))
        
        next_state = copy_game_state(game)
        try:
            if move == "pass":
                next_state.play_pass()
            else:
                next_state.play_move(move[0], move[1])
        except ValueError:
            continue
            
        score = alphabeta(next_state, depth - 1, alpha, beta, False, target_player)
        move_scores.append((move, score))
        alpha = max(alpha, score)
        
    progress_bar.progress(1.0)
    status_placeholder.empty()
    progress_bar.empty()
    
    if move_scores:
        move_scores.sort(key=lambda x: x[1], reverse=True)
        best_score = move_scores[0][1]
        best_candidates = [move for move, score in move_scores if abs(score - best_score) < 1e-7]
        ai_move = random.choice(best_candidates)
        
        # UI 연동용 글로벌 변수에 결정 정보 기록
        LAST_AI_DECISION["move"] = ai_move
        LAST_AI_DECISION["score"] = best_score
        LAST_AI_DECISION["depth"] = depth
    else:
        ai_move = "pass"
        
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
        
    log_state("D_AI_AFTER")
    st.session_state.preview_coord = None
    st.rerun()
