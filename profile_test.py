import sys
import traceback

try:
    import os
    import random
    import io
    import cProfile
    import pstats
    from contextlib import redirect_stdout

    # 프로젝트 경로 추가
    sys.path.append(r"C:\Users\User\source\repos\greatkingdomAI")

    from engine.game_state import GameState
    from engine.board import BLUE, ORANGE
    from ai.minimax import find_best_move, clear_transposition_table, reset_stats
except Exception as e:
    with open("error_output.txt", "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)
    sys.exit(1)

def play_fast_profile_game():
    clear_transposition_table()
    reset_stats()
    
    game = GameState()
    game.is_copy = False
    
    move_count = 0
    max_moves = 100 # 최대 수순 제한
    
    while not game.game_over and move_count < max_moves:
        if move_count == 0:
            from ai.minimax import get_legal_moves
            legal = [m for m in get_legal_moves(game) if m != "pass"]
            move = random.choice(legal)
        else:
            f = io.StringIO()
            with redirect_stdout(f):
                # 깊이를 2로 낮추어 수읽기 연산량을 95% 이상 감축
                move = find_best_move(game, depth=2)
                
        if move == "pass":
            game.play_pass()
        else:
            try:
                game.play_move(move[0], move[1])
            except ValueError:
                break
        move_count += 1
        
    print(f"Fast Profile Game completed in {move_count} moves.")

if __name__ == "__main__":
    try:
        # cProfile 연산 수행
        cProfile.run("play_fast_profile_game()", "profile.out")
        
        # 결과 정렬 및 텍스트 저장
        with open("profile_results.txt", "w", encoding="utf-8") as f:
            p = pstats.Stats("profile.out", stream=f)
            p.sort_stats("cumulative")
            p.print_stats(30)
            
        from engine.safe_groups import GET_EMPTY_REGIONS_CALL_COUNT
        with open("output_test.txt", "w", encoding="utf-8") as f:
            f.write(f"ACTUAL CALLS: {GET_EMPTY_REGIONS_CALL_COUNT}\n")
            f.write("Success without exceptions.\n")
        print("Profiling analysis completed successfully.")
    except Exception as e:
        with open("error_output.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        sys.exit(1)
