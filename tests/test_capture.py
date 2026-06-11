import unittest
from engine.board import BLUE, ORANGE
from engine.game_state import GameState


class TestCapture(unittest.TestCase):

    def test_single_stone_capture(self):
        # 1. 단일 돌 포위 테스트
        game = GameState()
        # BLUE 돌을 (2, 2)에 배치
        game.board.place(2, 2, BLUE)

        # ORANGE가 (2, 2) 주변을 3곳 포위
        game.board.place(1, 2, ORANGE)
        game.board.place(3, 2, ORANGE)
        game.board.place(2, 1, ORANGE)

        # 마지막 한 수를 두어 캡처 발생
        game.current_player = ORANGE
        result = game.play_move(2, 3)

        self.assertTrue(result)
        self.assertEqual(game.check_winner(), ORANGE)
        self.assertEqual(game.board.get(2, 2), 0)  # 돌이 제거되어 EMPTY 상태여야 함

    def test_multiple_stones_capture(self):
        # 2. 다중 돌 포위 테스트 (연결된 2개의 BLUE 돌)
        game = GameState()
        game.board.place(2, 2, BLUE)
        game.board.place(2, 3, BLUE)

        # ORANGE가 둘러싸기
        # (2, 2) 인접: (1, 2), (3, 2), (2, 1)
        # (2, 3) 인접: (1, 3), (3, 3), (2, 4)
        game.board.place(1, 2, ORANGE)
        game.board.place(3, 2, ORANGE)
        game.board.place(2, 1, ORANGE)
        game.board.place(1, 3, ORANGE)
        game.board.place(3, 3, ORANGE)

        # 마지막 활로에 착수하여 2개 돌 모두 캡처
        game.current_player = ORANGE
        result = game.play_move(2, 4)

        self.assertTrue(result)
        self.assertEqual(game.check_winner(), ORANGE)
        self.assertEqual(game.board.get(2, 2), 0)
        self.assertEqual(game.board.get(2, 3), 0)

    def test_edge_capture(self):
        # 3. 가장자리 포위 테스트
        # (0, 0) 모서리는 벽으로 막혀 있어 2개의 돌로 캡처 가능
        game = GameState()
        game.board.place(0, 0, BLUE)
        game.board.place(0, 1, ORANGE)

        game.current_player = ORANGE
        result = game.play_move(1, 0)

        self.assertTrue(result)
        self.assertEqual(game.check_winner(), ORANGE)
        self.assertEqual(game.board.get(0, 0), 0)

    def test_neutral_adjacent_capture(self):
        # 4. 중립 성 인접 포위 테스트
        # (4, 4)는 NEUTRAL(중립 성)으로 벽 판정
        # (4, 3)에 BLUE 배치 시, (4, 4)는 벽이므로 나머지 3방향만 막으면 캡처
        game = GameState()
        game.board.place(4, 3, BLUE)
        game.board.place(3, 3, ORANGE)
        game.board.place(5, 3, ORANGE)

        game.current_player = ORANGE
        result = game.play_move(4, 2)

        self.assertTrue(result)
        self.assertEqual(game.check_winner(), ORANGE)
        self.assertEqual(game.board.get(4, 3), 0)


if __name__ == "__main__":
    unittest.main()
