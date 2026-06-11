import random

# 보드 크기는 9x9 고정
SIZE = 9
# 보드 상태 종류: 0 (EMPTY), 1 (BLUE), 2 (ORANGE), 3 (NEUTRAL)
STATES = 4

# 재현 가능하도록 랜덤 시드 고정
rng = random.Random(2026)

# 81칸 x 4가지 상태에 대한 64비트 랜덤 해시 값 테이블
ZOBRIST_TABLE = [
    [
        [rng.getrandbits(64) for _ in range(STATES)]
        for _ in range(SIZE)
    ]
    for _ in range(SIZE)
]

# 플레이어 차례 구분용 64비트 랜덤 해시 값
ZOBRIST_PLAYER = rng.getrandbits(64)


def get_board_hash(board, current_player):
    """현재 보드 구성 상태와 플레이어 차례에 대한 Zobrist 64비트 정수 해시값을 계산합니다."""
    h = 0
    size = board.size
    for r in range(size):
        for c in range(size):
            val = board.get(r, c)
            h ^= ZOBRIST_TABLE[r][c][val]

    # BLUE 플레이어 차례인 경우에 해시를 XOR 적용하여 차례 전이를 인지시킴
    if current_player == 1:
        h ^= ZOBRIST_PLAYER

    return h
