import random

class AIPlayer:
    def __init__(self, player):
        self.player = player

    def play(self, game):
        """
        Get the AI bot's move.
        
        TODO: Implement AI algorithms (minimax search)

        """
        return self.get_random_move(game)
    
    def get_random_move(self, game):
        """
        Get a random move for the AI player. For testing.

        """
        valid_moves = game.get_valid_moves
        return random.choice(valid_moves) if valid_moves else None