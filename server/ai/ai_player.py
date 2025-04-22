import random
import utils
import copy
from ai import evaluator

class AIPlayer:
    def __init__(self, player):
        self.player = player

    def play(self, game):
        """
        Get the AI bot's move.
        
        TODO: Implement AI algorithms (minimax search)

        """
        return self.findBestMove(game.board_state)
    
    def get_random_move(self, game):
        """
        Get a random move for the AI player. For testing.

        """
        valid_moves = game.get_valid_moves
        print(evaluator.evaluate(game.board_state))
        return random.choice(valid_moves) if valid_moves else None
    
    def findBestMove(self, board_state):
        bestVal = -1000
        bestMove = None
        
        for move in utils.get_valid_moves(board_state, 2):
            # Sao chép board
            temp_board = copy.deepcopy(board_state)
            
            # Áp dụng nước đi lên temp_board
            row, col = move
            utils.make_move(temp_board, row, col, self.player)

            # Đánh giá trạng thái sau khi đi
            value = evaluator.minimax(temp_board, 0, True, -1000, 1000)

            if value > bestVal:
                bestVal = value
                bestMove = move

        return bestMove
