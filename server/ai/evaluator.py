import utils
import numpy as np
from othello import BLACK, WHITE

INFINITY = 100_000
CORNERS = [(0,0), (0,7), (7,0), (7,7)]

def evaluate(board_state, player):
    """
    Evaluate the board state for the given player.
    The evaluation is based on several heuristics:
    - Difference in number of discs
    - Corner ownership
    - Mobility
    - Stability
    - Positional score
    - Frontier discs
    Returns:
        total_score: A float representing the evaluation score for the player.
    """
    disc_diff_score = get_disc_diff(board_state, player)
    corner_diff_score = get_corner_diff(board_state, player)
    mobility_score = get_mobility(board_state, player)
    stability_score = get_stability(board_state, player)
    positional_score = get_positional_score(board_state, player)
    frontier_score = get_frontier_discs(board_state, player)

    # Combine scores with weights (adjust on testing)
    total_score = (disc_diff_score * 1.0 +
                   corner_diff_score * 10.0 +
                   mobility_score * 1.5 +
                   stability_score * 3.0 +
                   positional_score * 1.0 +
                   frontier_score * -1.0)

    return total_score

def minimax(board_state, depth, player, isMax, alpha, beta):
    """
    Minimax algorithm with alpha-beta pruning to evaluate the best move.
    Args:
        board_state: Current state of the board.
        depth: Depth of the search tree.
        isMax: True if it's the maximizing player's turn, False otherwise.
        alpha: Alpha value for pruning.
        beta: Beta value for pruning.
    Returns:
        score: The evaluation score for the current board state.
    """

    # If the game is over, return the evaluation score based on the winner
    if utils.is_game_over(board_state):
        winner = utils.get_winner(board_state)
        if winner == player:
            return INFINITY
        if winner == utils.get_opponent(player):
            return -INFINITY
        return 0
    
    # When reached the maximum depth, return the evaluation score
    if depth == 0:
        return evaluate(board_state, player)
    
     # Xác định player hiện tại
    if not isMax:
        player = utils.get_opponent(player)
    
    # Get valid moves for the current player
    valid_moves = utils.get_valid_moves(board_state, player)
    # Sort valid moves by their evaluation score
    valid_moves.sort(key=lambda move: evaluate(utils.make_move(board_state, move[0], move[1], player), player), reverse=isMax)

    if not valid_moves:
        # Mất lượt -> chuyển cho đối thủ nhưng giữ nguyên board
        return minimax(board_state, depth - 1, player, not isMax, alpha, beta)

    #If this is MAX
    if isMax:
        best = -INFINITY
        for move in valid_moves:
            row, col = move
            next_board = utils.make_move(board_state, row, col, player)
            val = minimax(next_board, depth - 1, player, not isMax, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = INFINITY
        for move in valid_moves:
            row, col = move
            next_board = utils.make_move(board_state, row, col, player)
            val = minimax(next_board, depth - 1, player, not isMax, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def get_disc_diff(board_state, player):
    """
    Calculate the difference in the number of discs between the player and the opponent.
    """
    opponent = utils.get_opponent(player)
    player_count = sum(row.count(player) for row in board_state)
    opponent_count = sum(row.count(opponent) for row in board_state)
    if (player_count + opponent_count) == 0:
        return 0
    return (player_count - opponent_count) / (player_count + opponent_count)


def get_corner_diff(board_state, player):
    """
    Calculate the difference in corner ownership between the player and the opponent.
    Corners are considered valuable positions in Othello.
    """
    opponent = utils.get_opponent(player)
    player_corners = sum(1 for corner in CORNERS if board_state[corner[0]][corner[1]] == player)
    opponent_corners = sum(1 for corner in CORNERS if board_state[corner[0]][corner[1]] == opponent)
    if player_corners + opponent_corners == 0:
        return 0
    return (player_corners - opponent_corners) / (player_corners + opponent_corners)


def get_mobility(board_state, player):
    """
    Calculate the mobility of the player.
    Mobility is the number of valid moves available to the player.
    """
    opponent = utils.get_opponent(player)
    player_moves = utils.count_valid_moves(board_state, player)
    opponent_moves = utils.count_valid_moves(board_state, opponent)
    if player_moves + opponent_moves == 0:
        return 0
    return (player_moves - opponent_moves) / (player_moves + opponent_moves)


def get_stability(board, player):
    n = 8
    stable = [[False for _ in range(n)] for _ in range(n)]

    # Mark corners and propagate along edges
    for r, c in CORNERS:
        if board[r][c] != 0:
            stable[r][c] = True
            # Vertical propagation
            if r == 7:
                rr = r
                while rr > 0:
                    rr -= 1
                    if board[rr][c] == board[rr+1][c]:
                        stable[rr][c] = True
                    else:
                        break
            else:
                rr = r
                while rr < 7:
                    rr += 1
                    if board[rr][c] == board[rr-1][c]:
                        stable[rr][c] = True
                    else:
                        break
            # Horizontal propagation
            if c == 7:
                cc = c
                while cc > 0:
                    cc -= 1
                    if board[r][cc] == board[r][cc+1]:
                        stable[r][cc] = True
                    else:
                        break
            else:
                cc = c
                while cc < 7:
                    cc += 1
                    if board[r][cc] == board[r][cc-1]:
                        stable[r][cc] = True
                    else:
                        break

    # Propagate stability inward
    changed = True
    while changed:
        changed = False
        for i in range(1, n-1):
            for j in range(1, n-1):
                if stable[i][j]:
                    continue
                if board[i][j] == 0:
                    continue
                # Check all 4 orthogonal and 4 diagonal directions
                if (
                    (board[i][j] == board[i+1][j] and stable[i+1][j]) and
                    (board[i][j] == board[i-1][j] and stable[i-1][j]) and
                    (board[i][j] == board[i][j+1] and stable[i][j+1]) and
                    (board[i][j] == board[i][j-1] and stable[i][j-1]) and
                    (board[i][j] == board[i+1][j+1] and stable[i+1][j+1]) and
                    (board[i][j] == board[i-1][j-1] and stable[i-1][j-1]) and
                    (board[i][j] == board[i+1][j-1] and stable[i+1][j-1]) and
                    (board[i][j] == board[i-1][j+1] and stable[i-1][j+1])
                ):
                    stable[i][j] = True
                    changed = True

    player_stable = sum(1 for i in range(n) for j in range(n) if board[i][j] == player and stable[i][j])
    opponent_stable = sum(1 for i in range(n) for j in range(n) if board[i][j] == utils.get_opponent(player) and stable[i][j])

    if player_stable + opponent_stable == 0:
        return 0
    return (player_stable - opponent_stable) / (player_stable + opponent_stable)


def get_positional_score(board_state, player):
    """
    Calculate the positional score based on the player's discs.
    Positional score is a heuristic that considers the positions of the discs on the board.
    """
    positional_weights = np.array([
        [100, -20, 10,  5,  5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [ 10,  -2,  3,  2,  2,  3,  -2,  10],
        [  5,  -2,  2,  1,  1,  2,  -2,   5],
        [  5,  -2,  2,  1,  1,  2,  -2,   5],
        [ 10,  -2,  3,  2,  2,  3,  -2,  10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10,  5,  5, 10, -20, 100]
    ])
    # If a corner is occupied, the cells adjacent to it are more valuable
    for corner in CORNERS:
        r, c = corner
        if board_state[r][c] == player:
            # Increase the value of adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if (dr != 0 or dc != 0) and utils.is_on_board(r + dr, c + dc):
                        positional_weights[r + dr][c + dc] += 50
    # Calculate the positional score
    score = 0
    for r in range(len(board_state)):
        for c in range(len(board_state[r])):
            if board_state[r][c] == player:
                score += positional_weights[r][c]
            elif board_state[r][c] == utils.get_opponent(player):
                score -= positional_weights[r][c]
    return score / 100


def get_frontier_discs(board_state, player):
    """
    Get the number of frontier discs for the player.
    Frontier discs are those that are adjacent to empty squares.
    """
    player_frontier = 0
    opponent_frontier = 0

    for r in range(8):
        for c in range(8):
            if board_state[r][c] == 0:
                continue
            is_frontier = False
            for dr, dc in utils.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if utils.is_on_board(nr, nc) and board_state[nr][nc] == 0:
                    is_frontier = True
                    break
            if is_frontier:
                if board_state[r][c] == player:
                    player_frontier += 1
                else:
                    opponent_frontier += 1

    if player_frontier + opponent_frontier == 0:
        return 0
    return (player_frontier - opponent_frontier) / (player_frontier + opponent_frontier)