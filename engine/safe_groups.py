from collections import deque
from engine.board import EMPTY, BLUE, ORANGE


DIRECTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


EMPTY_REGIONS_CACHE = {}


def clear_empty_regions_cache():
    EMPTY_REGIONS_CACHE.clear()


GET_EMPTY_REGIONS_CALL_COUNT = 0


def get_empty_regions(board):
    """보드의 모든 빈 영역(EMPTY)들을 BFS로 찾아 반환합니다.

    반환: list of (region_set, adjacent_colors_set, adjacent_stones_set, is_four_edge_bool)
    """
    global GET_EMPTY_REGIONS_CALL_COUNT
    GET_EMPTY_REGIONS_CALL_COUNT += 1

    board_tuple = tuple(tuple(row) for row in board.grid)
    if board_tuple in EMPTY_REGIONS_CACHE:
        return EMPTY_REGIONS_CACHE[board_tuple]

    size = board.size
    visited = set()
    regions = []

    for r in range(size):
        for c in range(size):
            if board.get(r, c) == EMPTY and (r, c) not in visited:
                region = set()
                q = deque([(r, c)])
                visited.add((r, c))

                adjacent_colors = set()
                adjacent_stones = set()

                touch_top = False
                touch_bottom = False
                touch_left = False
                touch_right = False

                while q:
                    cr, cc = q.popleft()
                    region.add((cr, cc))

                    if cr == 0:
                        touch_top = True
                    if cr == size - 1:
                        touch_bottom = True
                    if cc == 0:
                        touch_left = True
                    if cc == size - 1:
                        touch_right = True

                    for dr, dc in DIRECTIONS:
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
                            adjacent_stones.add((nr, nc))

                is_four_edge = (
                    touch_top and touch_bottom and touch_left and touch_right
                )
                regions.append(
                    (region, adjacent_colors, adjacent_stones, is_four_edge)
                )

    EMPTY_REGIONS_CACHE[board_tuple] = regions
    return regions



def is_safe_group(board, group, player, regions=None):
    """주어진 player의 돌 그룹(group)이 안전 그룹인지 판별합니다.

    해당 그룹이 이미 완성된 자기 영토(Four Edge Rule 미적용 영토)를 둘러싸고 있다면 True를 반환합니다.
    """
    if regions is None:
        regions = get_empty_regions(board)

    for region, adj_colors, adj_stones, is_four_edge in regions:
        # Four Edge Rule에 걸린 영토는 정식 영토가 아니므로 제외
        if is_four_edge:
            continue

        # 해당 영토가 player의 영토인지 확인
        if adj_colors == {player}:
            # 이 영토를 감싸고 있는 모든 돌들이 현재 group의 일부인지 확인
            if adj_stones.issubset(group):
                return True

    return False
