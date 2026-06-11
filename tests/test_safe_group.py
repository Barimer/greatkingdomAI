import unittest
from engine.board import BLUE, ORANGE
from engine.capture import get_group
from engine.game_state import GameState
from engine.safe_groups import is_safe_group


class TestSafeGroup(unittest.TestCase):

    def test_safe_group_detection(self):
        # 1. 완성 집 보유 그룹 감지 테스트
        game = GameState()
        # BLUE가 (0, 0)을 둘러싸 자기 영토(집)를 완성함
        game.board.place(0, 1, BLUE)
        game.board.place(1, 1, BLUE)
        game.board.place(1, 0, BLUE)

        # (1, 0) 좌표의 BLUE 돌과 연결된 그룹 추출
        group = get_group(game.board, 1, 0)

        # 이 그룹이 BLUE의 안전 그룹으로 정상 감지되는지 검증
        is_safe = is_safe_group(game.board, group, BLUE)
        self.assertTrue(is_safe)

    def test_safe_group_cannot_be_captured(self):
        # 2. 안전 그룹의 캡처 불가 검증 테스트
        game = GameState()
        # BLUE가 (0, 0)에 영토를 형성
        game.board.place(0, 1, BLUE)
        game.board.place(1, 1, BLUE)
        game.board.place(1, 0, BLUE)

        # ORANGE가 외부 활로인 (0, 2), (1, 2), (2, 1), (2, 0)을 모두 차단
        game.board.place(0, 2, ORANGE)
        game.board.place(1, 2, ORANGE)
        game.board.place(2, 1, ORANGE)
        game.board.place(2, 0, ORANGE)

        # 마지막 활로이자 내부 공간인 (0, 0)에 ORANGE가 착수
        # BLUE 그룹의 활로는 이로써 0이 되지만, 안전 그룹이므로 캡처되어서는 안 됨
        game.current_player = ORANGE
        result = game.play_move(0, 0)

        # 캡처 즉시 승리가 발생하지 않았으므로 play_move는 False를 리턴해야 함
        self.assertFalse(result)

        # BLUE 돌 그룹이 캡처되지 않고 보드 위에 정상 보존되는지 확인
        self.assertEqual(game.board.get(0, 1), BLUE)
        self.assertEqual(game.board.get(1, 1), BLUE)
        self.assertEqual(game.board.get(1, 0), BLUE)


if __name__ == "__main__":
    unittest.main()
