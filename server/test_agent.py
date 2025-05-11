'''
import arena
import othello

from ai.ai_player import AIPlayer, RandomPlayer

game = othello.Game()

player1 = AIPlayer(game).play
player2 = RandomPlayer(game).play

# Define number of games
num_games = 20

# Start the competition
#set_seed(seed=SEED)
#arena = arena.Arena(player1, player2, game, display=None)  # To see the steps of the competition set "display=OthelloGame.display"

arena = arena.Arena(player1)  # To see the steps of the competition set "display=OthelloGame.display"
arena.game = game
result = arena.playGames(num_games, verbose=False)  # returns (Number of player1 wins, number of player2 wins, number of ties)

# Compute win rate for the random player (player 1)
print(f"\nNumber of games won by player1 = {result[0]}, "
      f"Number of games won by player2 = {result[1]} out of {num_games} games")
win_rate_player1 = result[0]/num_games
print(f"\nWin rate for player1 over 20 games: {round(win_rate_player1*100, 1)}%")

'''
from test_enviroment import OthelloEnv  # Import môi trường Othello của bạn
from othello import BLACK, WHITE
from ai.ai_player import AIPlayer, RandomPlayer # Import các Player của bạn

if __name__ == "__main__":
    env = OthelloEnv()
    # Tạo các agent
    ai_agent_black = AIPlayer(BLACK)
    random_agent_white = RandomPlayer(env.game)  # Cần truyền game vào RandomPlayer

    #tạo vòng lặp 20 game để đánh giá thuật toán nào oke hơn
    # cho đoạn code dưới vào loop 20 
    #lười làm

    observation, info = env.reset()
    done = False
    current_player = BLACK

    while not done:
        if current_player == BLACK:
            action = ai_agent_black.play(env.game)
        else:
            action = random_agent_white.play(env.game)

        # Chuyển đổi định dạng nước đi của AIPlayer sang action_space của env
        if action is not None:
            row, col = action
            action_gym = row * env.board_size + col
        else:
            # Xử lý trường hợp không có nước đi hợp lệ (pass)
            action_gym = env.action_space.n - 1  # Giả định hành động cuối cùng là "pass" (nếu bạn xử lý pass)
            # Trong Othello, không có hành động "pass" rõ ràng, bạn có thể cần điều chỉnh logic này
            # để phù hợp với cách bạn xử lý "pass" trong code của mình.
            # Ví dụ: nếu không có nước đi hợp lệ, agent trước vẫn phải gọi env.step() với một action bất kì
            # và env.step() sẽ tự xử lý việc không có nước đi hợp lệ.

        next_observation, reward, terminated, truncated, info = env.step(action_gym)
        done = terminated or truncated
        observation = next_observation
        current_player = 3 - current_player  # Chuyển đổi giữa BLACK (1) và WHITE (2)
        env.render()

    print("Game Over!")
    if reward > 0:
        print("BLACK wins!")
    elif reward < 0:
        print("WHITE wins!")
    else:
        print("It's a draw!")

    env.close()