"""
TODO: Add evaluate function for AI bot.
The evaluate function should return a score based on the current game state.

"""
import copy
import utils
from othello import BLACK, WHITE

def evaluate(board_state):
    dif = utils.get_score(board_state, WHITE) - utils.get_score(board_state, BLACK)
    return dif

def minimax(board_state, depth, isMax):
    "MAX is WHITE, MIN is BLACK"
    score = evaluate(board_state);
    print("DEPTH =")
    print(depth)
    print(isMax)
    print(score)
    if not (utils.get_valid_moves(board_state, BLACK) and utils.get_valid_moves(board_state, WHITE)):
        if score == 0:
            return 0
        if score > 0:
            return 64
        return -64
    
    temp_board = copy.deepcopy(board_state)

    if (depth > 3):
        return score
    
    # If this maximizer's move  
    if isMax:
        best = -1000;
        for move in utils.get_valid_moves(board_state, BLACK):
            row, col = move
            temp_board = utils.make_move(temp_board, row, col, BLACK)
            best = max(best, minimax(temp_board, depth+1, not isMax));
        return best
    
    # If this minimizer's move  
    else:
        best = 1000;
        for move in utils.get_valid_moves(board_state, WHITE):
            row, col = move
            temp_board = utils.make_move(temp_board, row, col, WHITE)
            best = min(best, minimax(temp_board, depth+1, not isMax));
        return best
    
