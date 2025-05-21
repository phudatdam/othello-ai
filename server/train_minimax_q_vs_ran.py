import os
import torch
import numpy as np
from server.train.test_environment import OthelloEnv
from ai.ai_player import RandomPlayer
from minimax_q_learning import MinimaxQAgent
from q_learning import TrainingMetrics
WHITE = 2
BLACK = 1
MODEL_PATH = "models/minimax_q_network.pt"

if __name__ == "__main__":
    metrics = TrainingMetrics()
    total_black_win = 0
    total_white_win = 0
    total_draw = 0
    num_games = 1000

    agent_white = MinimaxQAgent()
    if os.path.exists(MODEL_PATH):
        agent_white.model.load_state_dict(torch.load(MODEL_PATH))
        agent_white.model.eval()
        print("✅ Model loaded from", MODEL_PATH)
    else:
        print("⚠️ No saved model found, training from scratch.")

    for game_index in range(num_games):
        env = OthelloEnv()
        random_agent_black = RandomPlayer(env.game)

        observation, info = env.reset()
        done = False
        current_player = BLACK

        state = agent_white.encode_state(observation)
        last_action = None

        while not done:
            if current_player == WHITE:
                # Encode current state
                state_tensor = agent_white.encode_state(observation)

                # Lấy các nước đi hợp lệ của WHITE (agent)
                valid_moves = env.game.get_valid_moves
                valid_idxs = [r * 8 + c for r, c in valid_moves]

                # Chọn hành động từ agent
                if valid_moves:
                    action = agent_white.choose_action(observation, valid_idxs)
                    row, col = divmod(action, env.board_size)
                else:
                    action = env.action_space.n - 1
                    row, col = None, None
            else:
                # BLACK là random player
                move = random_agent_black.play(env.game)
                if move is not None:
                    row, col = move
                    action = row * env.board_size + col
                else:
                    action = env.action_space.n - 1
                    row, col = None, None

                last_action = action  # Lưu phản ứng của BLACK để train sau

            # Thực hiện bước đi
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Encode trạng thái tiếp theo
            next_state_tensor = agent_white.encode_state(next_observation)

            if current_player == WHITE:
                # Tạo bản sao để tính toán next_valid_moves và next_opponent_moves_dict
                temp_game = env.game.deepcopy()
                temp_game.turn = WHITE  # Giả định WHITE đi tiếp tại s'

                next_valid_moves = [r * 8 + c for r, c in temp_game.get_valid_moves]
                next_opponent_moves_dict = {}

                for a in next_valid_moves:
                    tmp2 = temp_game.deepcopy()
                    r, c = divmod(a, env.board_size)
                    tmp2.play(tmp2.turn, r, c)  # WHITE đi
                    tmp2.turn = BLACK           # Đối thủ phản ứng

                    opp_moves = tmp2.get_valid_moves
                    if opp_moves:
                        opp_actions = [r * 8 + c for r, c in opp_moves]
                    else:
                        opp_actions = [env.action_space.n - 1]

                    next_opponent_moves_dict[a] = opp_actions

                # Train minimax Q-agent
                opponent_action = last_action if last_action is not None else agent_white.PASS_ACTION
                agent_white.train(
                    state_tensor=state_tensor,
                    action=action,
                    opponent_action=opponent_action,
                    reward=reward,
                    next_state_tensor=next_state_tensor,
                    done=done,
                    next_valid_moves=next_valid_moves,
                    next_opponent_moves_dict=next_opponent_moves_dict
                )

            # Cập nhật cho vòng lặp tiếp theo
            state = next_state_tensor
            observation = next_observation
            current_player = 3 - current_player  # Đổi lượt

        winner = env.game.get_winner()
        if winner == WHITE:
            total_white_win += 1
        elif winner == BLACK:
            total_black_win += 1
        else:
            total_draw += 1

        if agent_white.epsilon > agent_white.epsilon_min:
            agent_white.epsilon -= agent_white.epsilon_decay

        win_rate = total_white_win / (game_index + 1)
        metrics.update(
            win_rate=win_rate,
            loss=agent_white.current_loss,
            epsilon=agent_white.epsilon,
            reward=reward
        )

        if game_index % 50 == 0 and game_index > 0:
            metrics.plot()
            print(f"📈 Saved chart at training_metrics_{game_index}.png")

    metrics.plot("final_training_metrics.png")
    torch.save(agent_white.model.state_dict(), MODEL_PATH)
    print(f"💾 Saved model to {MODEL_PATH}")

    print("\n=== RESULT AFTER", num_games, "GAMES ===")
    print(f"WHITE (MinimaxQAgent) wins: {total_white_win}")
    print(f"BLACK (RandomPlayer) wins: {total_black_win}")
    print(f"Draws: {total_draw}")
