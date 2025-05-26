import random
from ai import minimax
import numpy as np
import torch
from network.q_learning import QNetwork, QNetworkAgent
import time

class MinimaxPlayer:
    def __init__(self, player):
        self.player = player

    def play(self, game):
        """
        Play a move using the Minimax algorithm with alpha-beta pruning.
        Args:
            game: The current game state.
        Returns:
            bestMove: The best move for the player in the form (row, col).
        """
        return self.find_best_move(game.board_state)
    
    def find_best_move(self, board_state):
        # start = time.time()
        max_depth = 4
        time_limit = 5
        best_move = minimax.id_minimax(board_state, self.player, max_depth, time_limit)
        # end = time.time()
        # print(f"Iterative deepening evaluation time: {end - start:.4f} seconds")
        return best_move
    

class RandomPlayer():

    def __init__(self, game):
        self.game = game

    def get_random_move(self, game):
        """
        Get a random move for the AI player. For testing.

        """
        valid_moves = game.get_valid_moves

        #print(evaluator.evaluate(game.board_state))
        return random.choice(valid_moves) if valid_moves else None

    def play(self, game):
        """
        Args:
        board: np.ndarray
            Board of size n x n [6x6 in this case]

        Returns:
            a: int
            Randomly chosen move
        """

        return self.get_random_move(game)
    

model_path = "models/q_network.pt"
class QLearningPlayer:
    def __init__(self, player):
        self.player = player
        self.board_size = 8

        # Load Q-network
        self.model = QNetwork();
        self.model.load_state_dict(torch.load(model_path))
        print("Model loaded from", model_path)
        self.model.eval()  # Set model to evaluation mode

    def get_state_tensor(self, board, turn):
        if turn == 2:
            board = np.where(board == 1, -1, board)
            board = np.where(board == 2, 1, board)
        else:
            board = np.where(board == 1, 1, board)
            board = np.where(board == 2, -1, board)
        return torch.tensor(board, dtype=torch.float32).reshape(-1)

    def play(self, game):
        valid_moves = game.get_valid_moves
        if not valid_moves:
            return None

        board = np.array(game.board_state)

        state_tensor = self.get_state_tensor(board, self.player)

        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()

        # Lọc Q-values theo hành động hợp lệ
        best_action = max(valid_moves, key=lambda a: q_values[a[0] * 8 + a[1]])
        return best_action

