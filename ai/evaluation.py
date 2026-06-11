from engine.board import EMPTY, BLUE, ORANGE
from engine.capture import get_group, get_liberties
from engine.territory import calculate_territory_details


def evaluate_detailed(board, player):
    """현재 보드 상태에서 각 평가 항목별 세부 가중치 점수 및 총점을 계산하여 반환합니다."""
    opponent = ORANGE if player == BLUE else BLUE
    size = board.size
    center = size // 2

    # 1. 영토 차이 (가중치 10.0)
    blue_t, orange_t, _, _ = calculate_territory_details(board)
    my_territory = blue_t if player == BLUE else orange_t
    opp_territory = orange_t if player == BLUE else blue_t
    territory_diff = my_territory - opp_territory
    territory_score = territory_diff * 10.0

    # 돌 분류
    my_stones = []
    opp_stones = []
    empty_cells = []

    for r in range(size):
        for c in range(size):
            val = board.get(r, c)
            if val == player:
                my_stones.append((r, c))
            elif val == opponent:
                opp_stones.append((r, c))
            elif val == EMPTY:
                empty_cells.append((r, c))

    # 2. 잠재 영토 (Influence, 가중치 1.5)
    my_influence = 0
    opp_influence = 0
    for r, c in empty_cells:
        my_adj = 0
        opp_adj = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if board.is_valid(nr, nc):
                    val = board.get(nr, nc)
                    if val == player:
                        my_adj += 1
                    elif val == opponent:
                        opp_adj += 1
        if my_adj > opp_adj:
            my_influence += 1
        elif opp_adj > my_adj:
            opp_influence += 1

    influence_diff = my_influence - opp_influence
    influence_score = influence_diff * 1.5

    # 3. 중앙 장악 (Center control, 가중치 0.8)
    my_center_score = 0
    opp_center_score = 0
    max_dist = size * 2
    for r, c in my_stones:
        dist = abs(r - center) + abs(c - center)
        my_center_score += max_dist - dist
    for r, c in opp_stones:
        dist = abs(r - center) + abs(c - center)
        opp_center_score += max_dist - dist

    center_diff = my_center_score - opp_center_score
    center_score = center_diff * 0.8

    # 4. 그룹 자유도 (가중치 2.0) & 5. 연결성 (가중치 0.5)
    # 추가: 내 그룹 위험 감점 및 상대 위협 가산점
    def analyze_groups_detailed(stones, is_opponent=False):
        visited = set()
        groups = []
        total_liberties = 0
        penalty_or_bonus = 0

        for r, c in stones:
            if (r, c) not in visited:
                group = get_group(board, r, c)
                visited.update(group)
                groups.append(group)
                libs = get_liberties(board, group)
                num_libs = len(libs)
                total_liberties += num_libs

                if not is_opponent:
                    # 내 그룹인 경우 자유도 위험 패널티 (지시서 기준 재조정)
                    if num_libs == 1:
                        penalty_or_bonus -= 150
                    elif num_libs == 2:
                        penalty_or_bonus -= 30
                    elif num_libs == 3:
                        penalty_or_bonus -= 5
                else:
                    # 상대 그룹인 경우 단수/공격 위협 보너스 (지시서 기준 완화 유지)
                    if num_libs <= 2:
                        penalty_or_bonus += 20


        connectivity = len(stones) - len(groups)
        return total_liberties, connectivity, penalty_or_bonus

    my_liberties, my_connectivity, my_danger_penalty = analyze_groups_detailed(my_stones, is_opponent=False)
    opp_liberties, opp_connectivity, opp_threat_bonus = analyze_groups_detailed(opp_stones, is_opponent=True)

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
