import random
import utils
import copy
from ai import evaluator
import numpy as np
import pickle
from collections import defaultdict
import torch
import os
import time
from network.q_learning import QNetwork, QNetworkAgent
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
        return self.find_best_move(game)
    
    def find_best_move(self, game):
        start = time.time()
        best_val = -evaluator.INFINITY - 1
        best_move = None
        # Lấy player hiện tại từ game
        player = self.player
        # Lấy nước đi hợp lệ từ game cho player này
        valid_moves = game.get_valid_moves
        for move in valid_moves:
            row, col = move
            board = copy.deepcopy(game.board_state)
            utils.make_move(board, row, col, player)
            depth = evaluator.get_search_depth(board)
            value = evaluator.minimax(board, depth, player, False, -evaluator.INFINITY, evaluator.INFINITY)
            print("Move của black:", move, "giá trị:", value)
            if value >= best_val:
                best_val = value
                best_move = move
        end = time.time()
        print(f"Minimax evaluation time: {end - start:.4f} seconds")
        print(f"Best move: {best_move} with value: {best_val}")
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
    
q_model_path = "models/q_network.pt"
class QLearningPlayer:
    def __init__(self, player):
        self.player = player
        self.board_size = 8

        # Load Q-network
        self.model = QNetwork();
        self.model.load_state_dict(torch.load(q_model_path))
        print("Model loaded from", q_model_path)
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

class MinimaxQLearningPlayer:
    def __init__(self, player):
        self.player = player
        self.board_size = 8
        from network.minimax_q_learning import MinimaxQAgent
        self.agent = MinimaxQAgent()
        # Load minimax Q-network nếu có
        mnm_q_model_path = "models/minimax_q_network.pt"
        if os.path.exists(mnm_q_model_path):
            self.agent.model.load_state_dict(torch.load(mnm_q_model_path))
            self.agent.model.eval()
            print("MinimaxQ model loaded from", mnm_q_model_path)
        else:
            print("No MinimaxQ model found, using random weights.")

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
        obs = {'board': board, 'turn': self.player}
        # Sử dụng agent để chọn action minimax Q
        action = self.agent.choose_exploitation_action(obs, valid_moves)
        return action

