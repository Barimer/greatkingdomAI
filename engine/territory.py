from collections import deque
from engine.board import EMPTY, BLUE, ORANGE

_TERRITORY_CACHE = {}


def calculate_territory_details(board):
    # board.grid의 튜플 상태를 캐싱 키로 사용 (성능 폭증)
    grid_tuple = tuple(tuple(row) for row in board.grid)
    if grid_tuple in _TERRITORY_CACHE:
        return _TERRITORY_CACHE[grid_tuple]

    size = board.size
    visited = set()
    blue_score = 0
    orange_score = 0
    blue_coords = []
    orange_coords = []

    for r in range(size):
        for c in range(size):
            if board.get(r, c) == EMPTY and (r, c) not in visited:
                region = []
                q = deque([(r, c)])
                visited.add((r, c))

                adjacent_colors = set()
                touch_top = False
                touch_bottom = False
                touch_left = False
                touch_right = False

                while q:
                    cr, cc = q.popleft()
                    region.append((cr, cc))

                    if cr == 0:
                        touch_top = True
                    if cr == size - 1:
                        touch_bottom = True
                    if cc == 0:
                        touch_left = True
                    if cc == size - 1:
                        touch_right = True

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr = cr + dr
                        nc = cc + dc

                        if not board.is_valid(nr, nc):
                            continue

                        val = board.get(nr, nc)
                        if val == EMPTY:
                            if (nr, nc) not in visited:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                        elif val in (BLUE, ORANGE):
                            adjacent_colors.add(val)

                # Four Edge Rule: 위, 아래, 왼쪽, 오른쪽 모든 변에 동시에 접촉한 경우 영토 미인정
                is_four_edge = (
                    touch_top and touch_bottom and touch_left and touch_right
                )

                if not is_four_edge:
                    if adjacent_colors == {BLUE}:
                        blue_score += len(region)
                        blue_coords.extend(region)
                    elif adjacent_colors == {ORANGE}:
                        orange_score += len(region)
                        orange_coords.extend(region)

    # 오름차순 정렬하여 보기 쉽게 반환
    blue_coords.sort()
    orange_coords.sort()

    res = blue_score, orange_score, blue_coords, orange_coords
    _TERRITORY_CACHE[grid_tuple] = res
    return res


def calculate_territory(board):
    blue_score, orange_score, _, _ = calculate_territory_details(board)
    return blue_score, orange_score
