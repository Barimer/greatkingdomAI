print("=== GREAT KINGDOM AI ===")
print("Select Game Mode:")
print("1. Human (BLUE) vs AI (ORANGE)")
print("2. AI (BLUE) vs Human (ORANGE)")
print("3. Human (BLUE) vs Human (ORANGE)")

# 모드 선택 검증 루프
while True:
    mode_input = input("Choose mode (1-3): ").strip()
    if mode_input in ("1", "2", "3"):
        break
    print("Invalid option. Please choose 1, 2, or 3.")

from engine.board import BLUE, ORANGE
from engine.game_state import GameState
from engine.territory import calculate_territory, calculate_territory_details
from ai.minimax import find_best_move

# 각 플레이어별 컨트롤러 맵핑
controllers = {}
if mode_input == "1":
    controllers[BLUE] = "Human"
    controllers[ORANGE] = "AI"
elif mode_input == "2":
    controllers[BLUE] = "AI"
    controllers[ORANGE] = "Human"
else:
    controllers[BLUE] = "Human"
    controllers[ORANGE] = "Human"

print(f"\nGame Start: {controllers[BLUE]} vs {controllers[ORANGE]}")
game = GameState()

while True:
    # 보드 및 상태 표시
    game.board.display()

    # 실시간 현재 영토 표시
    blue_t, orange_t = calculate_territory(game.board)
    print(f"Current Territory -> BLUE: {blue_t} | ORANGE: {orange_t}")

    # 현재 차례 플레이어 종류 및 정보 출력
    current_player_str = (
        "BLUE (Player 1)" if game.current_player == BLUE else "ORANGE (Player 2)"
    )
    current_type = controllers[game.current_player]
    print(f"Current Turn: {current_player_str} [{current_type}]")

    result = False

    # 1. AI 차례인 경우
    if current_type == "AI":
        print("AI is calculating...")
        move = find_best_move(game, depth=2)
        print(f"AI Decision: {move}")

        if move == "pass":
            result = game.play_pass()
        else:
            result = game.play_move(move[0], move[1])

    # 2. Human 차례인 경우
    else:
        move = input("r c (or 'pass', 'score') : ")
        clean_move = move.strip().lower()

        if clean_move == "pass":
            result = game.play_pass()
        elif clean_move == "score":
            (
                blue_score,
                orange_score,
                blue_coords,
                orange_coords,
            ) = calculate_territory_details(game.board)
            print()
            print(f"BLUE Territory : {blue_score}")
            print(f"BLUE Territory Coordinates: {blue_coords}")
            print(f"ORANGE Territory : {orange_score}")
            print(f"ORANGE Territory Coordinates: {orange_coords}")
            print()
            continue
        else:
            try:
                r, c = map(
                    int,
                    move.split()
                )
                result = game.play_move(r, c)
            except ValueError as e:
                print(
                    "Invalid input. Please enter 'r c' coordinates, 'pass', or 'score'."
                )
                print("Details:", e)
                continue

    # 승리 혹은 종료 이벤트 발생 시 루프 중단
    if result:
        game.board.display()
        winner = game.check_winner()
        winner_str = (
            "BLUE (Player 1)" if winner == BLUE else "ORANGE (Player 2)"
        )
        print(f"\nGame Over! Winner is {winner_str}")
        break