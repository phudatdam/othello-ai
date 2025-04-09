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
def is_on_board(r, c):
    return 0 <= r < 8 and 0 <= c < 8

def is_valid_move(board_state, player, row, col):
    if (board_state[row][col] != EMPTY):
        return False
    
    opponent = 3 - player

    for dr, dc in DIRECTIONS:
        temp_row, temp_col = row + dr, col + dc
        can_flip = False
        while is_on_board(temp_row, temp_col) and board_state[temp_row][temp_col] == opponent:
            temp_row += dr
            temp_col += dc
            can_flip = True
        if can_flip and is_on_board(temp_row, temp_col) and board_state[temp_row][temp_col] == player:
            return True
        
    return False

def get_valid_moves(board_state, player):
    valid_moves = [(row, col) for row in range(8) for col in range(8) if is_valid_move(board_state, player, row, col)]
    return valid_moves

def make_move(board_state, row, col, player):
    board_state[row][col] = player
    opponent = 3 - player
    for dr, dc in DIRECTIONS:
        temp_row, temp_col = row + dr, col + dc
        pieces_to_flip = []
        while is_on_board(temp_row, temp_col) and board_state[temp_row][temp_col] == opponent:
            pieces_to_flip.append((temp_row, temp_col))
            temp_row += dr
            temp_col += dc
        if pieces_to_flip and is_on_board(temp_row, temp_col) and board_state[temp_row][temp_col] == player:
            for i, j in pieces_to_flip:
                board_state[i][j] = player

    return board_state

def flip_pieces(board_state, row, col, player):
    pass

def is_game_over(board_state):
    return not get_valid_moves(board_state, 1) and not get_valid_moves(board_state, 2)

def get_score(board_state, player):
    score = 0
    for row in range(8):
        for col in range(8):
            if board_state[row][col] == player:
                score += 1
    return score

def get_winner(board_state):
    black_score = get_score(board_state, BLACK)
    white_score = get_score(board_state, WHITE)
    if black_score > white_score:
        return BLACK
    elif white_score > black_score:
        return WHITE
    else:
        return None