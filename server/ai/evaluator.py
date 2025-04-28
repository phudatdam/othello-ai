"""
TODO: Add evaluate function for AI bot.
The evaluate function should return a score based on the current game state.

"""
import pandas as pd
import copy
import utils
import numpy as np
from data_extractor import extract_features
from othello import BLACK, WHITE

CORNERS = [(0,0), (0,7), (7,0), (7,7)]

MANUAL_WEIGHTS = np.array([
    [-1.3, 17.3, 94.1, -1.3],   # Phase 0
    [-2.3, 18.7, 92.7, -0.14],   # Phase 1
    [1.9, 21.9, 90.6, 5.7],   # Phase 2
    [3, 24.8, 82.1, 8.5],   # Phase 3
    [3, 21.5, 55.7, 14.4],   # Phase 4
    [8.9, 0, 0, 65.9],  # Phase 5
])

def evaluate(board_state, player):
    move_count = sum(1 for i in range(8) for j in range(8) if board_state[i][j] != 0)
    phase = determine_phase(move_count)
    features = extract_features(board_state, player)
    
    # Lấy weights thủ công theo phase
    weights = MANUAL_WEIGHTS[phase]
    
    # Tính điểm không cần chuẩn hóa (vì weights đã được scale thủ công)
    raw_score = np.dot(features, weights)
    
    return raw_score # Điều chỉnh hệ số theo thực nghiệm

def determine_phase(move_count):
    """Calculate game phase based on move count"""
    if move_count < 10:
        return 0
    return min((move_count - 10) // 10, 5)

def minimax(board_state, depth, isMax, alpha, beta):
    "MAX is WHITE, MIN is BLACK"
    score = evaluate(board_state, WHITE if isMax else BLACK);
    
    print("DEPTH =")
    print(depth)
    print(isMax)
    print(score)
    
    
    
    if not (utils.get_valid_moves(board_state, BLACK) and utils.get_valid_moves(board_state, WHITE)):
        if score == 0:
            return 0
        if score > 0:
            return 1000
        return -1000
    
    temp_board = copy.deepcopy(board_state)

    if (depth >= 4 ):
       return score
    
     # Xác định player hiện tại
    player = WHITE if isMax else BLACK
    valid_moves = utils.get_valid_moves(board_state, player)
    valid_moves = sorted(valid_moves, key=lambda move: move in CORNERS, reverse=True)
    if not valid_moves:
        # Mất lượt -> chuyển cho đối thủ nhưng giữ nguyên board
        return minimax(board_state, depth + 1, not isMax, alpha, beta)

    #If this is MAX
    if isMax:
        best = -1000
        for move in valid_moves:
            row, col = move
            temp_board = copy.deepcopy(board_state)
            temp_board = utils.make_move(temp_board, row, col, player)
            val = minimax(temp_board, depth+1, not isMax, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = 1000
        for move in valid_moves:
            row, col = move
            temp_board = copy.deepcopy(board_state)
            temp_board = utils.make_move(temp_board, row, col, player)
            val = minimax(temp_board, depth+1, not isMax, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best