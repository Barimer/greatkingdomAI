from engine.board import (
    BLUE,
    ORANGE
)

from engine.capture import (
    get_group,
    get_liberties
)

from engine.safe_groups import is_safe_group


class GameState:

    def __init__(self):

        from engine.board import Board

        self.board = Board()

        self.current_player = BLUE
        self.consecutive_passes = 0
        self.winner = None
        self.game_over = False
        self.is_copy = False  # 시뮬레이션 여부 판단용 기본값

    def opponent(self):

        if self.current_player == BLUE:
            return ORANGE

        return BLUE

    def play_move(self, r, c):

        if self.game_over:
            raise ValueError("Game is already over")

        opponent = self.opponent()

        # 착수하기 전 보드 상태에서 상대방의 안전 그룹을 미리 수집
        safe_stones = set()
        checked_pre = set()

        for rr in range(self.board.size):
            for cc in range(self.board.size):
                if self.board.get(rr, cc) != opponent:
                    continue
                if (rr, cc) in checked_pre:
                    continue

                group = get_group(self.board, rr, cc)
                checked_pre.update(group)

                if is_safe_group(self.board, group, opponent):
                    safe_stones.update(group)

        # 이제 착수를 수행합니다.
        self.board.place(
            r,
            c,
            self.current_player
        )

        # 착수를 성공했으므로 연속 패스 초기화
        self.consecutive_passes = 0

        captured = []
        checked = set()

        for rr in range(self.board.size):
            for cc in range(self.board.size):
                if self.board.get(rr, cc) != opponent:
                    continue
                if (rr, cc) in checked:
                    continue

                group = get_group(
                    self.board,
                    rr,
                    cc
                )
                checked.update(group)

                # 착수 전에 안전 그룹에 해당했던 돌들은 캡처 대상에서 영구 배제
                if group.issubset(safe_stones):
                    continue

                libs = get_liberties(
                    self.board,
                    group
                )

                if len(libs) == 0:
                    captured.extend(group)

        if captured:

            for pr, pc in captured:
                self.board.remove(pr, pc)

            # [개선] 가상 시뮬레이션 복사본이 아닐 때만 콘솔에 캡처 발생 출력
            if not getattr(self, "is_copy", False):
                print()
                print("CAPTURE OCCURRED")
                print(captured)
                print()

            self.winner = self.current_player
            self.game_over = True
            return True

        self.current_player = opponent

        return False

    def play_pass(self):

        if self.game_over:
            raise ValueError("Game is already over")

        self.consecutive_passes += 1

        if self.consecutive_passes >= 2:
            self.game_over = True
            self.check_winner_by_territory()
            return True

        self.current_player = self.opponent()

        return False

    def check_winner_by_territory(self):

        from engine.territory import calculate_territory

        blue, orange = calculate_territory(self.board)

        if blue >= orange + 3:
            self.winner = BLUE
        else:
            self.winner = ORANGE

    def check_winner(self):

        return self.winner