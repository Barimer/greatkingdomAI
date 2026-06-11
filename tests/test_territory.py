import unittest
from engine.board import Board, BLUE, EMPTY
from engine.territory import calculate_territory


class TestTerritory(unittest.TestCase):

    def test_normal_territory(self):
        # 1. 일반 영토 테스트
        # 5x5 보드에서 (0, 0)을 BLUE 돌 3개와 가장자리 벽으로 둘러싸 영토 1점 획득 검증
        board = Board(size=5)
        board.place(0, 1, BLUE)
        board.place(1, 1, BLUE)
        board.place(1, 0, BLUE)

        blue_score, orange_score = calculate_territory(board)
        self.assertEqual(blue_score, 1)
        self.assertEqual(orange_score, 0)

    def test_neutral_castle_wall(self):
        # 2. 중립 성 활용 테스트
        # 5x5 보드에서 (2, 2)는 NEUTRAL(중립 성)
        # (1, 2) 빈 칸을 (0, 2), (1, 1), (1, 3)의 BLUE 돌로 막고 아래쪽은 중립 성을 벽으로 활용
        board = Board(size=5)
        board.place(0, 2, BLUE)
        board.place(1, 1, BLUE)
        board.place(1, 3, BLUE)

        blue_score, orange_score = calculate_territory(board)
        self.assertEqual(blue_score, 1)
        self.assertEqual(orange_score, 0)

    def test_board_edge_wall(self):
        # 3. 가장자리 활용 테스트
        # 3x3 보드에서 (0, 0) 빈 칸을 (0, 1)과 (1, 0)의 BLUE 돌로 막아 가장자리를 벽으로 활용
        board = Board(size=3)
        board.place(0, 1, BLUE)
        board.place(1, 0, BLUE)

        blue_score, orange_score = calculate_territory(board)
        self.assertEqual(blue_score, 1)
        self.assertEqual(orange_score, 0)

    def test_four_edge_rule(self):
        # 4. Four Edge Rule 테스트

        # Case A: 영토 불인정 (4변 모두에 접촉)
        # 3x3 보드에서 중심(1, 1)에 BLUE 배치. 테두리 8칸의 빈 공간이 상하좌우 모든 가장자리에 닿아 있음.
        board_a = Board(size=3)
        board_a.grid[1][1] = BLUE  # 강제로 중심을 BLUE로 변경 (기본 중립성 덮어씀)

        blue_score_a, orange_score_a = calculate_territory(board_a)
        self.assertEqual(blue_score_a, 0)  # Four Edge Rule에 의해 0점이어야 함
        self.assertEqual(orange_score_a, 0)

        # Case B: 영토 인정 (3변에만 접촉, 아래쪽 가장자리에는 닿지 않음)
        # 3x3 보드에서 (2,0), (2,1), (2,2), (1,0), (1,2)를 BLUE로 배치
        # 빈 공간 영역은 {(0,0), (0,1), (0,2), (1,1)} -> 총 4칸
        board_b = Board(size=3)
        board_b.grid[1][1] = EMPTY  # 중립 성 비우기
        board_b.place(2, 0, BLUE)
        board_b.place(2, 1, BLUE)
        board_b.place(2, 2, BLUE)
        board_b.place(1, 0, BLUE)
        board_b.place(1, 2, BLUE)

        blue_score_b, orange_score_b = calculate_territory(board_b)
        self.assertEqual(blue_score_b, 4)  # 4변 전부에 닿지 않았으므로 정상적으로 4점 획득
        self.assertEqual(orange_score_b, 0)


if __name__ == "__main__":
    unittest.main()
