import pandas as pd
import numpy as np
import utils
from othello import BLACK, WHITE, Game

# Định nghĩa các cột và hàng trên bàn cờ
columns = 'abcdefgh'
rows = '12345678'

CORNERS = [(0,0), (0,7), (7,0), (7,7)]

def move_to_coords(move):
    """Chuyển nước đi từ dạng 'e6' sang tọa độ (row, col)"""
    if len(move) != 2:
        return None
    col = columns.find(move[0])
    row = rows.find(move[1])
    if col == -1 or row == -1:
        return None
    return row, col

def extract_features(board, player):
    player_score = utils.get_score(board, player)
    opponent_score = utils.get_score(board, 3 - player)
    features = [
        (player_score - opponent_score)  / (player_score + opponent_score),
        corner_diff(board, player),
        mobility(board, player),
        frontier_discs(board, player),
        stability(board, player),
        player_score + opponent_score
    ]
    return features

def corner_diff(board, player):
    player_corners = 0
    opponent_corners = 0
    
    for r, c in CORNERS:
        if board[r][c] == player:
            player_corners += 1
        elif board[r][c] == 3 - player:
            opponent_corners += 1
        # Nếu board[r][c] == 0 => bỏ qua

    if player_corners + opponent_corners == 0: return 0  # Không ai chiếm góc nào => 0
    return (player_corners - opponent_corners)  / (player_corners + opponent_corners)

def coin_parity(board, player):
    player_score = utils.get_score(board, player)
    opponent_score = utils.get_score(board, 3 - player)

    return (player_score - opponent_score)  / (player_score + opponent_score)

def mobility(board, player):
    player_move_counts = utils.count_valid_moves(board, player)
    opponent_move_counts = utils.count_valid_moves(board, 3 - player)
    if player_move_counts + opponent_move_counts == 0: return 0
    return (player_move_counts - opponent_move_counts)  / (player_move_counts + opponent_move_counts)

def frontier_discs(board, player):
    opponent = 3 - player
    my_frontier = 0
    opponent_frontier = 0

    for r in range(8):
        for c in range(8):
            if board[r][c] == 0:
                continue
            is_frontier = False
            for dr, dc in utils.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == 0:
                    is_frontier = True
                    break
            if is_frontier:
                if board[r][c] == player:
                    my_frontier += 1
                else:
                    opponent_frontier += 1

    if my_frontier + opponent_frontier == 0:
        return 0
    return (my_frontier - opponent_frontier)  / (my_frontier + opponent_frontier)

def stability(board, player):
    n = 8
    stable = [[False for _ in range(8)] for _ in range(8)]  # Ban đầu chưa ô nào stable

    # Gán ổn định ban đầu từ các ô ở biên
    for r,c in CORNERS:
        if board[r][c] != 0:
            stable[r][c] = True
            if r == 7:
                while r > 0:
                    r -= 1
                    if board[r][c] == board[r+1][c]:
                        stable[r][c] = True
            else:
                while r < 7:
                    r += 1
                    if board[r][c] == board[r-1][c]:
                        stable[r][c] = True
            if c == 7:
                while c > 0:
                    c -= 1
                    if board[r][c] == board[r][c+1]:
                        stable[r][c] = True
            else:
                while c < 7:
                    c += 1
                    if board[r][c] == board[r][c-1]:
                        stable[r][c] = True
    # Xét các ô ở giữa
    changed = True
    while changed:
        changed = False
        for i in range(1, n-1):
            for j in range(1, n-1):
                if stable[i][j]:
                    continue  # Ô này không phải quân mình hoặc đã stable rồi
                count = 0
                if (board[i][j] != board[i+1][j] or not stable[i+1][j]) and (board[i][j] != board[i-1][j] or not stable[i-1][j]):
                    continue
                if (board[i][j] != board[i][j+1] or not stable[i][j+1]) and (board[i][j] != board[i][j+1] or not stable[i][j+1]):
                    continue
                if (board[i][j] != board[i+1][j+1] or not stable[i+1][j+1]) and (board[i][j] != board[i-1][j-1] or not stable[i-1][j-1]):
                    continue
                if (board[i][j] != board[i+1][j-1] or not stable[i+1][j-1]) and (board[i][j] != board[i-1][j+1] or not stable[i-1][j+1]):
                    continue
                # Nếu cả 4 hướng đều kề stable thì ô này stable
                stable[i][j] = True
                changed = True
    # Đếm số lượng stable của player và oplayer
    oplayer = 3 - player
    my_stable = sum(1 for i in range(n) for j in range(n) if board[i][j] == player and stable[i][j])
    op_stable = sum(1 for i in range(n) for j in range(n) if board[i][j] == oplayer and stable[i][j])

    return (my_stable - op_stable)  / (my_stable + op_stable) if my_stable + op_stable != 0 else 0


if __name__ == "__main__":
    df = pd.read_csv('othello_dataset.csv')
    features = []
    labels = []

    for index, row in df.iterrows():
        game = Game()
        moves = [row['game_moves'][i:i+2] for i in range(0, len(row['game_moves']), 2)]
        for move in moves:
            coords = move_to_coords(move)
            if coords:
                r, c = coords
                game.board_state = utils.make_move(game.board_state, r, c, game.turn)
                if utils.get_valid_moves(game.board_state, 3 - game.turn):
                    game.turn = 3 - game.turn
                feat = extract_features(game.board_state, WHITE)
                features.append(feat)
                labels.append(row['winner'])

    features_df = pd.DataFrame(features, columns=[
        'score_diff','corner_diff', 'mobility', 'frontier_discs', 'stability','phase'
    ])
    features_df['label'] = labels
    features_df.to_csv('othello_features.csv', index=False)