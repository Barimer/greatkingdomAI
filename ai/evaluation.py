from engine.board import EMPTY, BLUE, ORANGE

# 9x9 보드 1차원 이웃 노드 테이블 및 가장자리 조건 사전 정의 (모듈 레벨 캐싱)
NEIGHBORS = {}
for r in range(9):
    for c in range(9):
        idx = r * 9 + c
        nb = []
        if r > 0: nb.append(idx - 9)
        if r < 8: nb.append(idx + 9)
        if c > 0: nb.append(idx - 1)
        if c < 8: nb.append(idx + 1)
        NEIGHBORS[idx] = nb

NEIGHBORS_8 = {}
for r in range(9):
    for c in range(9):
        idx = r * 9 + c
        nb = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    nb.append(nr * 9 + nc)
        NEIGHBORS_8[idx] = nb

IS_TOP = [idx // 9 == 0 for idx in range(81)]
IS_BOTTOM = [idx // 9 == 8 for idx in range(81)]
IS_LEFT = [idx % 9 == 0 for idx in range(81)]
IS_RIGHT = [idx % 9 == 8 for idx in range(81)]

def calculate_territory_flat(grid_flat):
    visited = [False] * 81
    blue_score = 0
    orange_score = 0

    for idx in range(81):
        if grid_flat[idx] == 0 and not visited[idx]:
            region_size = 0
            q = [idx]
            visited[idx] = True
            
            adjacent_colors = set()
            touch_top = False
            touch_bottom = False
            touch_left = False
            touch_right = False
            
            head = 0
            while head < len(q):
                curr = q[head]
                head += 1
                region_size += 1
                
                if IS_TOP[curr]: touch_top = True
                elif IS_BOTTOM[curr]: touch_bottom = True
                if IS_LEFT[curr]: touch_left = True
                elif IS_RIGHT[curr]: touch_right = True
                
                for n_idx in NEIGHBORS[curr]:
                    val = grid_flat[n_idx]
                    if val == 0:
                        if not visited[n_idx]:
                            visited[n_idx] = True
                            q.append(n_idx)
                    elif val == 1 or val == 2:
                        adjacent_colors.add(val)
                        
            is_four_edge = touch_top and touch_bottom and touch_left and touch_right
            if not is_four_edge:
                if len(adjacent_colors) == 1:
                    color = next(iter(adjacent_colors))
                    if color == 1:
                        blue_score += region_size
                    elif color == 2:
                        orange_score += region_size
                        
    return blue_score, orange_score

def analyze_groups_detailed_flat(grid_flat, stones, is_opponent=False):
    visited = [False] * 81
    total_liberties = 0
    penalty_or_bonus = 0
    num_groups = 0
    
    stone_indices = [r * 9 + c for r, c in stones]
    
    for idx in stone_indices:
        if visited[idx]:
            continue
            
        num_groups += 1
        group = [idx]
        visited[idx] = True
        
        liberties = set()
        
        head = 0
        while head < len(group):
            curr = group[head]
            head += 1
            
            for n_idx in NEIGHBORS[curr]:
                val = grid_flat[n_idx]
                if val == grid_flat[idx]:
                    if not visited[n_idx]:
                        visited[n_idx] = True
                        group.append(n_idx)
                elif val == 0:
                    liberties.add(n_idx)
                    
        num_libs = len(liberties)
        total_liberties += num_libs
        
        if not is_opponent:
            if num_libs == 1:
                penalty_or_bonus -= 150
            elif num_libs == 2:
                penalty_or_bonus -= 30
            elif num_libs == 3:
                penalty_or_bonus -= 5
        else:
            if num_libs <= 2:
                penalty_or_bonus += 20
                
    connectivity = len(stones) - num_groups
    return total_liberties, connectivity, penalty_or_bonus

def evaluate_detailed(board, player):
    """현재 보드 상태에서 각 평가 항목별 세부 가중치 점수 및 총점을 계산하여 반환합니다."""
    opponent = ORANGE if player == BLUE else BLUE
    size = 9
    center = 4

    # 평탄화 그리드 생성 (매우 빠름)
    grid_flat = board.grid[0] + board.grid[1] + board.grid[2] + board.grid[3] + board.grid[4] + board.grid[5] + board.grid[6] + board.grid[7] + board.grid[8]

    # 1. 영토 차이 (가중치 10.0)
    blue_t, orange_t = calculate_territory_flat(grid_flat)
    my_territory = blue_t if player == BLUE else orange_t
    opp_territory = orange_t if player == BLUE else blue_t
    territory_diff = my_territory - opp_territory
    territory_score = territory_diff * 10.0

    # 돌 분류
    my_stones = []
    opp_stones = []
    empty_cells = []

    for idx in range(81):
        val = grid_flat[idx]
        r = idx // 9
        c = idx % 9
        if val == player:
            my_stones.append((r, c))
        elif val == opponent:
            opp_stones.append((r, c))
        elif val == EMPTY:
            empty_cells.append((r, c))

    # 2. 잠재 영토 (Influence, 가중치 1.5)
    my_adj_grid = [0] * 81
    opp_adj_grid = [0] * 81

    for r, c in my_stones:
        idx = r * 9 + c
        for n_idx in NEIGHBORS_8[idx]:
            my_adj_grid[n_idx] += 1

    for r, c in opp_stones:
        idx = r * 9 + c
        for n_idx in NEIGHBORS_8[idx]:
            opp_adj_grid[n_idx] += 1

    my_influence = 0
    opp_influence = 0
    for r, c in empty_cells:
        idx = r * 9 + c
        my_adj = my_adj_grid[idx]
        opp_adj = opp_adj_grid[idx]
        if my_adj > opp_adj:
            my_influence += 1
        elif opp_adj > my_adj:
            opp_influence += 1

    influence_diff = my_influence - opp_influence
    influence_score = influence_diff * 1.5

    # 3. 중앙 장악 (Center control, 가중치 0.8)
    my_center_score = 0
    opp_center_score = 0
    max_dist = 18
    for r, c in my_stones:
        dist = abs(r - center) + abs(c - center)
        my_center_score += max_dist - dist
    for r, c in opp_stones:
        dist = abs(r - center) + abs(c - center)
        opp_center_score += max_dist - dist

    center_diff = my_center_score - opp_center_score
    center_score = center_diff * 0.8

    # 4. 그룹 자유도 (가중치 2.0) & 5. 연결성 (가중치 0.5)
    my_liberties, my_connectivity, my_danger_penalty = analyze_groups_detailed_flat(grid_flat, my_stones, is_opponent=False)
    opp_liberties, opp_connectivity, opp_threat_bonus = analyze_groups_detailed_flat(grid_flat, opp_stones, is_opponent=True)

    liberties_diff = my_liberties - opp_liberties
    connectivity_diff = my_connectivity - opp_connectivity

    liberties_score = liberties_diff * 2.0
    connectivity_score = connectivity_diff * 0.5

    # 최종 총합 점수
    total_score = (
        territory_score
        + influence_score
        + liberties_score
        + connectivity_score
        + center_score
        + my_danger_penalty      # 내 위험 감점 반영 (음수)
        + opp_threat_bonus       # 상대 공격 보너스 반영 (양수)
    )

    return {
        "Territory": territory_score,
        "Liberty": liberties_score,
        "Connectivity": connectivity_score,
        "Center": center_score,
        "Influence": influence_score,
        "Total": total_score,
    }

def evaluate(board, player):
    """단순 총점만 반환하는 기존 호환성용 평가 래퍼 함수입니다."""
    details = evaluate_detailed(board, player)
    return details["Total"]
