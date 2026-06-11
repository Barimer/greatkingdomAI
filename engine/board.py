EMPTY = 0
BLUE = 1
ORANGE = 2
NEUTRAL = 3


class Board:

    def __init__(self, size=9):

        self.size = size

        self.grid = [
            [EMPTY for _ in range(size)]
            for _ in range(size)
        ]

        center = size // 2

        self.grid[center][center] = NEUTRAL

    def is_valid(self, r, c):

        return (
            0 <= r < self.size
            and
            0 <= c < self.size
        )

    def get(self, r, c):

        return self.grid[r][c]

    def place(self, r, c, player):

        if not self.is_valid(r, c):
            raise ValueError(
                "invalid coordinate"
            )

        if self.grid[r][c] != EMPTY:
            raise ValueError(
                "occupied"
            )

        self.grid[r][c] = player

    def remove(self, r, c):

        self.grid[r][c] = EMPTY

    def copy(self):

        new_board = Board.__new__(Board)
        new_board.size = self.size
        new_board.grid = [
            row[:] for row in self.grid
        ]

        return new_board

    def display(self):

        symbols = {
            EMPTY: ".",
            BLUE: "B",
            ORANGE: "O",
            NEUTRAL: "N"
        }

        print()
        # 상단 열 번호 출력
        col_headers = "  " + " ".join(str(i) for i in range(self.size))
        print(col_headers)

        # 좌측 행 번호와 함께 행 출력
        for i, row in enumerate(self.grid):
            row_str = " ".join(
                symbols[cell]
                for cell in row
            )
            print(f"{i} {row_str}")

        print()

