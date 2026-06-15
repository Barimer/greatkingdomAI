import sys
import os

# 모듈 경로 추가
sys.path.append(os.path.abspath("."))

from engine.game_state import GameState
from engine.board import BLUE, ORANGE, EMPTY
from engine.territory import calculate_territory_details
from ai.minimax import get_legal_moves

def run_reproduction_test():
    print("=== [규칙 반영 검증 테스트] 영토 소유권 및 착수 제한 ===")
    
    # ------------------ CASE 1: BLUE가 완전히 둘러싼 영토 ------------------
    print("\n--- CASE 1: BLUE가 완전히 둘러싼 영토 ---")
    game1 = GameState()
    board1 = game1.board
    
    # (2,2) 주위의 8개 좌표에 BLUE 돌 배치
    blue_boundary = [
        (1, 1), (1, 2), (1, 3),
        (2, 1),         (2, 3),
        (3, 1), (3, 2), (3, 3)
    ]
    for r, c in blue_boundary:
        board1.place(r, c, BLUE)
            
    # 영토 세부 정보 가져오기
    _, _, blue_coords1, _ = calculate_territory_details(board1)
    is_blue_territory = (2, 2) in blue_coords1
    
    # ORANGE 합법 수 확인
    game1.current_player = ORANGE
    legal_moves1 = get_legal_moves(game1)
    is_orange_legal = (2, 2) in legal_moves1
    
    print(f"Territory Owner = {'BLUE' if is_blue_territory else 'None'}")
    print(f"ORANGE Legal Move = {is_orange_legal}")
    
    # 착수 시도 예외 검증
    try:
        game1.play_move(2, 2)
        print("결과: 착수 성공 (오류 - 제한 실패)")
    except ValueError as e:
        print(f"결과: 착수 실패 ({e}) - [정상]")

    # ------------------ CASE 2: ORANGE가 완전히 둘러싼 영토 ------------------
    print("\n--- CASE 2: ORANGE가 완전히 둘러싼 영토 ---")
    game2 = GameState()
    board2 = game2.board
    
    # (2,2) 주위의 8개 좌표에 ORANGE 돌 배치
    orange_boundary = [
        (1, 1), (1, 2), (1, 3),
        (2, 1),         (2, 3),
        (3, 1), (3, 2), (3, 3)
    ]
    for r, c in orange_boundary:
        board2.place(r, c, ORANGE)
            
    # 영토 세부 정보 가져오기
    _, _, _, orange_coords2 = calculate_territory_details(board2)
    is_orange_territory = (2, 2) in orange_coords2
    
    # BLUE 합법 수 확인
    game2.current_player = BLUE
    legal_moves2 = get_legal_moves(game2)
    is_blue_legal = (2, 2) in legal_moves2
    
    print(f"Territory Owner = {'ORANGE' if is_orange_territory else 'None'}")
    print(f"BLUE Legal Move = {is_blue_legal}")
    
    # 착수 시도 예외 검증
    try:
        game2.play_move(2, 2)
        print("결과: 착수 성공 (오류 - 제한 실패)")
    except ValueError as e:
        print(f"결과: 착수 실패 ({e}) - [정상]")

    # ------------------ CASE 3: 중립 영역 ------------------
    print("\n--- CASE 3: 중립 영역 ---")
    game3 = GameState()
    board3 = game3.board
    
    # (2,2)의 한쪽은 BLUE, 한쪽은 ORANGE로 둘러싸서 중립 상태 생성
    board3.place(1, 2, BLUE)
    board3.place(3, 2, ORANGE)
    
    # 영토 세부 정보 가져오기
    _, _, blue_coords3, orange_coords3 = calculate_territory_details(board3)
    is_blue_territory_neutral = (2, 2) in blue_coords3
    is_orange_territory_neutral = (2, 2) in orange_coords3
    
    # 양쪽 플레이어 합법 수 확인
    game3.current_player = BLUE
    is_blue_legal_neutral = (2, 2) in get_legal_moves(game3)
    
    game3.current_player = ORANGE
    is_orange_legal_neutral = (2, 2) in get_legal_moves(game3)
    
    print(f"Territory Owner = {'BLUE' if is_blue_territory_neutral else ('ORANGE' if is_orange_territory_neutral else 'NEUTRAL')}")
    print(f"BLUE Legal Move = {is_blue_legal_neutral}")
    print(f"ORANGE Legal Move = {is_orange_legal_neutral}")
    
    print("\n=== 검증 완료 ===")

if __name__ == "__main__":
    run_reproduction_test()
