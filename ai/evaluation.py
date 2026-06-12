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

# 미래 활로 위험 분석용 글로벌 캐시 및 거리 테이블
FUTURE_LIBERTY_CACHE = {}
ENABLE_FLR = True

def clear_future_liberty_cache():
    """매 수 결정 전에 미래 활로 캐시를 초기화합니다."""
    FUTURE_LIBERTY_CACHE.clear()

MANHATTAN_DIST = {}
for i in range(81):
    r1, c1 = i // 9, i % 9
    dist_list = []
    for j in range(81):
        r2, c2 = j // 9, j % 9
        dist_list.append(abs(r1 - r2) + abs(c1 - c2))
    MANHATTAN_DIST[i] = dist_list

def get_local_moves_flat(grid_flat, group, distance=2):
    local = set()
    for idx in group:
        dist_list = MANHATTAN_DIST[idx]
        for i in range(81):
            if grid_flat[i] == 0 and dist_list[i] <= distance:
                local.add(i)
    return list(local)

def play_move_flat(grid_flat, idx, player):
    next_grid = list(grid_flat)
    next_grid[idx] = player
    opponent = 3 - player
    visited = [False] * 81
    for i in range(81):
        if next_grid[i] == opponent and not visited[i]:
            group = [i]
            visited[i] = True
            liberties = set()
            head = 0
            while head < len(group):
                curr = group[head]
                head += 1
                for n in NEIGHBORS[curr]:
                    val = next_grid[n]
                    if val == opponent:
                        if not visited[n]:
                            visited[n] = True
                            group.append(n)
                    elif val == 0:
                        liberties.add(n)
            if len(liberties) == 0:
                for g_idx in group:
                    next_grid[g_idx] = 0
    return next_grid

def get_local_state_key(grid_flat, group):
    local_indices = set(group)
    for idx in group:
        dist_list = MANHATTAN_DIST[idx]
        for i in range(81):
            if dist_list[i] <= 2:
                local_indices.add(i)
    return tuple((i, grid_flat[i]) for i in sorted(list(local_indices)))

def future_lib_minimax_flat(grid_flat, depth, alpha, beta, is_maximizing, group_for_tracking, player):
    local_key = get_local_state_key(grid_flat, group_for_tracking)
    cache_key = (local_key, depth, is_maximizing, group_for_tracking, player)
    if cache_key in FUTURE_LIBERTY_CACHE:
        return FUTURE_LIBERTY_CACHE[cache_key]

    active = [idx for idx in group_for_tracking if grid_flat[idx] == player]
    if not active:
        return 0 # Captured
        
    visited = [False] * 81
    curr_idx = active[0]
    group = [curr_idx]
    visited[curr_idx] = True
    liberties = set()
    head = 0
    while head < len(group):
        curr = group[head]
        head += 1
        for n in NEIGHBORS[curr]:
            val = grid_flat[n]
            if val == player:
                if not visited[n]:
                    visited[n] = True
                    group.append(n)
            elif val == 0:
                liberties.add(n)
                
    if depth == 0:
        val = len(liberties)
        FUTURE_LIBERTY_CACHE[cache_key] = val
        return val

    local_moves = get_local_moves_flat(grid_flat, group, distance=2)
    local_moves.append(-1)
    
    opponent = 3 - player

    if is_maximizing:
        max_val = -float('inf')
        for mv in local_moves:
            if mv == -1:
                val = future_lib_minimax_flat(grid_flat, depth - 1, alpha, beta, False, group_for_tracking, player)
            else:
                next_grid = play_move_flat(grid_flat, mv, player)
                next_tracked = group_for_tracking
                is_adj = False
                for g_idx in group:
                    if mv in NEIGHBORS[g_idx]:
                        is_adj = True
                        break
                if is_adj:
                    next_tracked = next_tracked.union([mv])
                val = future_lib_minimax_flat(next_grid, depth - 1, alpha, beta, False, next_tracked, player)
            max_val = max(max_val, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        FUTURE_LIBERTY_CACHE[cache_key] = max_val
        return max_val
    else:
        min_val = float('inf')
        for mv in local_moves:
            if mv == -1:
                val = future_lib_minimax_flat(grid_flat, depth - 1, alpha, beta, True, group_for_tracking, player)
            else:
                next_grid = play_move_flat(grid_flat, mv, opponent)
                val = future_lib_minimax_flat(next_grid, depth - 1, alpha, beta, True, group_for_tracking, player)
            min_val = min(min_val, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        FUTURE_LIBERTY_CACHE[cache_key] = min_val
        return min_val

def get_future_liberty_risk_flat(grid_flat, group_for_tracking, player, num_libs):
    if num_libs == 1:
        return 0, 0, 0
    elif num_libs == 2:
        fl1 = future_lib_minimax_flat(grid_flat, 1, -float('inf'), float('inf'), False, group_for_tracking, player)
        fl2 = future_lib_minimax_flat(grid_flat, 3, -float('inf'), float('inf'), False, group_for_tracking, player)
        if fl2 <= 1:
            fl3 = future_lib_minimax_flat(grid_flat, 5, -float('inf'), float('inf'), False, group_for_tracking, player)
        else:
            fl3 = fl2
        return fl1, fl2, fl3
    else: # num_libs == 3
        fl1 = future_lib_minimax_flat(grid_flat, 1, -float('inf'), float('inf'), False, group_for_tracking, player)
        fl2 = future_lib_minimax_flat(grid_flat, 3, -float('inf'), float('inf'), False, group_for_tracking, player)
        fl3 = fl2
        return fl1, fl2, fl3

_FLAT_TERRITORY_CACHE = {}

def calculate_territory_flat(grid_flat):
    grid_tuple = tuple(grid_flat)
    if grid_tuple in _FLAT_TERRITORY_CACHE:
        return _FLAT_TERRITORY_CACHE[grid_tuple]

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
                        
    res = blue_score, orange_score
    _FLAT_TERRITORY_CACHE[grid_tuple] = res
    return res

def analyze_groups_detailed_flat(grid_flat, stones, player, is_opponent=False):
    visited = [False] * 81
    total_liberties = 0
    penalty_or_bonus = 0
    num_groups = 0
    min_liberty = 99  # 기본값 (돌이 없을 때)
    min_liberty_fl3 = 99  # 기본값 (대마의 fl3 위험 지표)
    num_atari_groups = 0
    min_liberty_coords = set()
    
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
        
        # 미래 활로 위험(Future Liberty Risk) 패널티 및 보너스 계산
        fl3_val = 3
        if ENABLE_FLR and num_libs <= 3:
            owner = 3 - player if is_opponent else player
            group_set = frozenset(group)
            fl1, fl2, fl3 = get_future_liberty_risk_flat(grid_flat, group_set, owner, num_libs)
            fl3_val = fl3
            
            group_size = len(group)
            if not is_opponent:
                future_risk_penalty = 0.0
                if fl3 == 0:
                    future_risk_penalty = -50000.0 - 10000.0 * group_size
                elif fl3 == 1:
                    future_risk_penalty = -10000.0 - 2000.0 * group_size
                elif fl3 == 2:
                    future_risk_penalty = -1000.0 - 200.0 * group_size
                penalty_or_bonus += future_risk_penalty
            else:
                future_attack_bonus = 0.0
                if fl3 == 0:
                    future_attack_bonus = 40000.0 + 8000.0 * group_size
                elif fl3 == 1:
                    future_attack_bonus = 8000.0 + 1600.0 * group_size
                elif fl3 == 2:
                    future_attack_bonus = 800.0 + 160.0 * group_size
                penalty_or_bonus += future_attack_bonus
        
        if num_libs < min_liberty:
            min_liberty = num_libs
            min_liberty_coords = liberties.copy()
            min_liberty_fl3 = fl3_val
        elif num_libs == min_liberty:
            min_liberty_coords.update(liberties)
            min_liberty_fl3 = min(min_liberty_fl3, fl3_val)
        
        if num_libs == 1:
            num_atari_groups += 1
        
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
    return total_liberties, connectivity, penalty_or_bonus, min_liberty, num_atari_groups, min_liberty_coords, min_liberty_fl3

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

    # 3. 중앙 장악 (Center control, 가중치 0.0으로 비활성화)
    center_score = 0.0

    # 4. 그룹 자유도 (가중치 2.0) & 5. 연결성 (가중치 0.5)
    my_liberties, my_connectivity, my_danger_penalty, my_min_liberty, my_atari_groups, my_min_lib_coords_raw, my_min_lib_fl3 = analyze_groups_detailed_flat(grid_flat, my_stones, player, is_opponent=False)
    opp_liberties, opp_connectivity, opp_threat_bonus, opp_min_liberty, opp_atari_groups, opp_min_lib_coords_raw, opp_min_lib_fl3 = analyze_groups_detailed_flat(grid_flat, opp_stones, player, is_opponent=True)

    liberties_diff = my_liberties - opp_liberties
    connectivity_diff = my_connectivity - opp_connectivity

    liberties_score = liberties_diff * 2.0
    connectivity_score = connectivity_diff * 0.5

    # 1차원 인덱스 세트를 (r,c) 좌표 세트로 변환
    my_min_liberty_coords = {(idx // 9, idx % 9) for idx in my_min_lib_coords_raw}
    opp_min_liberty_coords = {(idx // 9, idx % 9) for idx in opp_min_lib_coords_raw}

    # 수정 2: Emergency Mode 적용
    if my_min_liberty == 1:
        territory_score = 0.0
        influence_score = 0.0
        center_score = 0.0
        connectivity_score = 0.0

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

    # 수정 4: 게임오버 위험 강제 패널티 적용
    if my_min_liberty == 1:
        total_score -= 10000.0
    elif my_min_liberty == 0:
        total_score -= 1000000.0

    if opp_min_liberty == 1:
        total_score += 5000.0

    # 양단수 (Double Atari) 보너스 적용
    double_atari_bonus = 0.0
    if opp_atari_groups >= 2:
        double_atari_bonus = 15000.0
        total_score += double_atari_bonus

    return {
        "Territory": territory_score,
        "Liberty": liberties_score,
        "Connectivity": connectivity_score,
        "Center": center_score,
        "Influence": influence_score,
        "my_min_liberty": my_min_liberty,
        "opp_min_liberty": opp_min_liberty,
        "opp_atari_groups": opp_atari_groups,
        "double_atari_bonus": double_atari_bonus,
        "my_min_liberty_coords": my_min_liberty_coords,
        "opp_min_liberty_coords": opp_min_liberty_coords,
        "my_min_lib_fl3": my_min_lib_fl3,
        "opp_min_lib_fl3": opp_min_lib_fl3,
        "my_danger_penalty": my_danger_penalty,
        "opp_threat_bonus": opp_threat_bonus,
        "Total": total_score,
    }

def evaluate(board, player):
    """단순 총점만 반환하는 기존 호환성용 평가 래퍼 함수입니다."""
    details = evaluate_detailed(board, player)
    return details["Total"]
