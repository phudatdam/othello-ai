import utils

EMPTY = 0
BLACK = 1
WHITE = 2


class Game:
    """
    An Othello game. Handles the game state and rules.

    Play moves with :meth:`play`.

    Get the current state of the game with :attr:`board_state`.

    Get the current turn with :attr:`turn`.

    Check for a victory with :attr:`winner`.

    """

    def __init__(self):
        self.board_state = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 1, 0, 0, 0],
            [0, 0, 0, 1, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
        self.turn = BLACK
        self.winner = None

    def get_valid_moves(self):
        return utils.get_valid_moves(self.board_state, self.turn)

    def play(self, player, row, col):
        """
        Play a move in a column.

        Returns the row where the checker lands.

        Raises :exc:`ValueError` if the move is illegal.

        """
        if player != self.turn:
            raise ValueError("It isn't your turn.")
        # Game logic here
        print("You clicked")
        self.board_state = utils.make_move(self.board_state, row, col, player)
        self.turn = WHITE if player == BLACK else BLACK