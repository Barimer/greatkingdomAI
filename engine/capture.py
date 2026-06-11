from collections import deque

from engine.board import EMPTY


DIRECTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


def get_group(board, r, c):

    color = board.get(r, c)

    visited = set()

    q = deque()

    q.append((r, c))

    visited.add((r, c))

    while q:

        cr, cc = q.popleft()

        for dr, dc in DIRECTIONS:

            nr = cr + dr
            nc = cc + dc

            if not board.is_valid(nr, nc):
                continue

            if (nr, nc) in visited:
                continue

            if board.get(nr, nc) == color:

                visited.add((nr, nc))
                q.append((nr, nc))

    return visited


def get_liberties(board, group):

    liberties = set()

    for r, c in group:

        for dr, dc in DIRECTIONS:

            nr = r + dr
            nc = c + dc

            if not board.is_valid(nr, nc):
                continue

            if board.get(nr, nc) == EMPTY:
                liberties.add((nr, nc))

    return liberties
