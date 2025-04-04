EMPTY = 0
BLACK = 1
WHITE = 2

DIRECTIONS = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 1],
    [1, -1], [1, 0], [1, 1]
];

"""
TODO: Add helper functions for Othello.

"""

def is_valid_move(board_state, player, row, col):
    if (board_state[row][col] != EMPTY):
        return False
    return True

def get_valid_moves(board_state, player):
    valid_moves = [(row, col) for row in range(8) for col in range(8) if is_valid_move(board_state, player, row, col)]
    return valid_moves

def make_move(board_state, row, col, player):
    return board_state

def can_flip(board_state, row, col, player):
    return False

def flip_pieces(board_state, row, col, player):
    pass

def is_game_over(board_state):
    return False

def get_score(board_state, player):
    return 0

def get_winner(board_state):
    return None