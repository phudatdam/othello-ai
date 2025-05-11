import pandas as pd
import numpy as np
import utils
from othello import BLACK, WHITE, Game
import torch

df = pd.read_csv('C:/Users/THE ANH/Desktop/code/prj_game_ai/othello-ai/server/data/othello_dataset.csv')

columns = 'abcdefgh'
rows = '12345678'


def move_to_coords(move):
    """Chuyển nước đi từ dạng 'e6' sang tọa độ (row, col)"""
    if len(move) != 2:
        return None
    col = columns.find(move[0])
    row = rows.find(move[1])
    if col == -1 or row == -1:
        return None
    return row, col



def othello_moves_to_tensor(moves_str):
    """Converts a string of Othello moves into a tensor representation of the final board state.

    Args:
        moves_str: A string representing the sequence of moves in Othello.
                   Each move is represented by two characters (e.g., 'f5', 'd6').

    Returns:
        A torch.Tensor of shape (1, 8, 8) representing the final board state.
        1 indicates black, -1 indicates white, and 0 indicates an empty cell.
    """
    board = torch.zeros(8, 8, dtype=torch.float32)
    player = 1  # Black starts

    def get_coords(move):
        col = ord(move[0]) - ord('a')
        row = int(move[1]) - 1
        return row, col

    for move in [moves_str[i:i+2] for i in range(0, len(moves_str), 2)]:
        if len(move) == 2:
            row, col = get_coords(move)
            if 0 <= row < 8 and 0 <= col < 8:
                board[row, col] = player
                #hoán đổi player
                if (player==1): 
                    player =2
                else:
                    player = 1

    return board.unsqueeze(0)


if __name__ == "__main__":
    df = pd.read_csv('C:/Users/THE ANH/Desktop/code/prj_game_ai/othello-ai/server/data/othello_dataset.csv')

    processed_data = []
    
    for index, row in df.iterrows():
        game = Game()
        #game_moves = row['game_moves']
        #board_state = othello_moves_to_tensor(game_moves)
        
        
        moves = [row['game_moves'][i:i+2] for i in range(0, len(row['game_moves']), 2)]
        for move in moves:
            coords = move_to_coords(move)
            if coords:
                r, c = coords
                game.board_state = utils.make_move(game.board_state, r, c, game.turn)
                if utils.get_valid_moves(game.board_state, 3 - game.turn):
                    game.turn = 3 - game.turn        

        board_state123 = torch.from_numpy(np.array(game.board_state))
        
        game_id = row['eOthello_game_id']
        winner = row['winner']
        processed_data.append({
            'game_id': game_id,
            'winner': winner,
            'board_state': board_state123
        })
        
    new_data_df = pd.DataFrame(processed_data, columns=[
        'game_id', 'winner', 'board_state'
    ])
    
    new_data_df.to_csv('othello_tensor_datasets.csv', index=False)

    