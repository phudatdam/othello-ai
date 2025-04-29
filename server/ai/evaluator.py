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

weights = pd.read_csv('phase_weights.csv', header=None).to_numpy()

def evaluate(board_state, player):
    raw_features = extract_features(board_state, player)
    phase = raw_features[5]
    features = raw_features[:-1]
    weight_index = phase
    phase_weights = weights[weight_index]
    score = np.dot(features, phase_weights)
    return score

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