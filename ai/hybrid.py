import torch
import numpy as np
import random
from engine.game_state import GameState
from engine.board import BLUE, ORANGE
from ai.minimax import get_legal_moves, copy_game_state, alphabeta
from model_v2 import PolicyNetworkV2

def board_to_tensor(board, current_player):
    opponent = 2 if current_player == 1 else 1
    state = np.zeros((4, 9, 9), dtype=np.float32)
    for r in range(9):
        for c in range(9):
            val = board.get(r, c)
            if val == current_player:
                state[0, r, c] = 1.0
            elif val == opponent:
                state[1, r, c] = 1.0
            elif val == 3:
                state[2, r, c] = 1.0
            else:
                state[3, r, c] = 1.0
    return state

def get_move_idx(move):
    if move == "pass" or (isinstance(move, (list, tuple, np.ndarray)) and move[0] == -1 and move[1] == -1):
        return 81
    return int(move[0] * 9 + move[1])

def get_move_from_idx(idx):
    if idx == 81:
        return "pass"
    return [idx // 9, idx % 9]

def find_hybrid_move(game_state, policy_model, device="cpu", temperature=None, top_k=5):
    """
    Hybrid AI:
    1. Policy V2 모델로부터 82개 행동 확률 출력
    2. 합법수 필터링 후 상위 K개 후보(Top-K) 선정
    3. temperature가 제공되지 않거나 0이면 결정론적 실전 대국 모드 (Threshold 검사 및 Argmax)
    4. 후보들에 대해 Depth 1 Minimax 평가 후 Soft Fusion Score 계산
    5. temperature가 지정된 경우 Temperature Softmax를 취해 확률 샘플링
    """
    if game_state.game_over:
        return "pass"
        
    curr_player = game_state.current_player
    legal_moves = get_legal_moves(game_state)
    if not legal_moves:
        return "pass"
        
    # 1. Policy Network 추론
    state_np = board_to_tensor(game_state.board, curr_player)
    state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = policy_model(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # 2. 합법수들의 확률 매핑 및 Top-K 추출
    move_probs = []
    for m in legal_moves:
        idx = get_move_idx(m)
        prob = probs[idx]
        move_probs.append((m, prob))
        
    # 확률 높은 순으로 정렬
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 K개 후보 선택
    top_candidates = move_probs[:top_k]
    
    # 3. 결정론적 모드 시 즉시 선택 (Threshold)
    if temperature is None or temperature == 0.0:
        if len(top_candidates) >= 2:
            p1 = top_candidates[0][1]
            p2 = top_candidates[1][1]
            if p1 >= 0.50 or (p1 / (p2 + 1e-5) >= 3.0):
                return top_candidates[0][0]
        elif len(top_candidates) == 1:
            return top_candidates[0][0]
        
    # 4. 각 후보에 대해 Depth 1 Minimax 평가
    candidate_scores = []
    for move, prob in top_candidates:
        next_state = copy_game_state(game_state)
        try:
            if move == "pass":
                next_state.play_pass()
            else:
                next_state.play_move(move[0], move[1])
        except ValueError:
            continue
            
        # Depth 1 Minimax 평가
        score = alphabeta(
            next_state, depth=0, alpha=-float("inf"), beta=float("inf"), 
            maximizing_player=False, target_player=curr_player
        )
        
        # Soft Fusion Score = Minimax Score + 30.0 * ln(prob)
        hybrid_score = score + 30.0 * np.log(prob + 1e-9)
        candidate_scores.append((move, hybrid_score))
        
    if not candidate_scores:
        return "pass"
        
    # 결정론적 모드 시 최종 최대 점수 선택 (Argmax)
    if temperature is None or temperature == 0.0:
        candidate_scores.sort(key=lambda x: -x[1])
        return candidate_scores[0][0]
        
    # 5. 확률적 Temperature Sampling 모드
    scores = np.array([cs[1] for cs in candidate_scores], dtype=np.float64)
    # Overflow 방지
    scores -= np.max(scores)
    
    # Softmax with temperature
    exp_scores = np.exp(scores / temperature)
    sum_exp = np.sum(exp_scores)
    if sum_exp > 0:
        sample_probs = exp_scores / sum_exp
    else:
        sample_probs = np.ones(len(candidate_scores)) / len(candidate_scores)
        
    chosen_idx = np.random.choice(len(candidate_scores), p=sample_probs)
    return candidate_scores[chosen_idx][0]
