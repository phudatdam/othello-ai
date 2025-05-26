import os
import torch
import numpy as np
from test_environment import OthelloEnv
from ai.ai_player import RandomPlayer
from network.minimax_q_learning import MinimaxQAgent
from network.metrics import TrainingMetrics
import utils

PASS_ACTION = -9
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
        agent_white.model.eval()  # Đặt model về chế độ đánh giá
        print("✅ Model loaded from", MODEL_PATH)
    else:
        print("⚠️ No saved model found, training from scratch.")

    for game_index in range(num_games):
        print(f"\n=== GAME {game_index + 1} ===")
        env = OthelloEnv()
        random_agent_black = RandomPlayer(env.game)

        observation, info = env.reset()
        #print("Trạng thái ban đầu:", env.game.turn)
        done = False
        current_player = BLACK

        # Biến tạm lưu trạng thái và action của White
        prev_state = None
        white_action = None
        total_reward = 0
        count = 0
        while not done:
            print("In ra count để biết vòng lặp có bị vô hạn không:", count)
            count += 1
            # === PHASE 1: BLACK'S TURN ===
            valid_moves_black = env.game.get_valid_moves
            if not valid_moves_black:
                # Nếu BLACK không có nước đi, kiểm tra WHITE
                valid_moves_white = env.game.get_valid_moves if env.game.turn == WHITE else utils.get_valid_moves(env.game.board_state, WHITE)
                if not valid_moves_white:
                    print("Cả hai bên đều không còn nước đi. Kết thúc game!")
                    break
            move = random_agent_black.play(env.game)
            if move is not None:
                row, col = move
            else:
                row, col = -1, -1  # Pass
            action_flat = row * env.board_size + col
            next_observation, reward, terminated, truncated, info = env.step(action_flat)
            done = terminated or truncated
            if prev_state is not None:
                next_state = agent_white.encode_state(next_observation)
                temp_reward = reward if current_player == WHITE else -reward
                agent_white.replay_buffer_save(
                    state=prev_state,
                    action=white_action,
                    opponent_action=(row, col),
                    reward=temp_reward,
                    next_state=next_state,
                    done=done
                )
            total_reward += reward
            observation = next_observation
            #env.render()
            if done:
                break
            # === PHASE 2: WHITE'S TURN ===
            valid_moves_white = env.game.get_valid_moves
            if not valid_moves_white:
                # Nếu WHITE không có nước đi, kiểm tra BLACK
                valid_moves_black = env.game.get_valid_moves if env.game.turn == BLACK else utils.get_valid_moves(env.game.board_state, BLACK)
                if not valid_moves_black:
                    print("Cả hai bên đều không còn nước đi. Kết thúc game!")
                    break
            prev_state = agent_white.encode_state(observation)
            if valid_moves_white:
                action = agent_white.choose_action(observation, valid_moves_white)
                row, col = action
            else:
                row, col = -1, -1  # Pass
            action_flat = row * env.board_size + col
            next_observation, _, _, _, _ = env.step(action_flat)
            white_action = (row, col)
            done = terminated or truncated
            observation = next_observation
            #env.render()
                
        # === KẾT THÚC MỘT GAME ===
        
        # Xử lý kết quả
        winner = env.game.get_winner()
        if winner == WHITE:
            total_white_win += 1
        elif winner == BLACK:
            total_black_win += 1
        else:
            total_draw += 1

        # Cập nhật epsilon và metrics

        metrics.update(
            win_rate=total_white_win/(game_index+1),
            loss=agent_white.current_loss,
            epsilon=agent_white.epsilon,
            reward=total_reward
        )

        # Train định kỳ
        if len(agent_white.memory) > 32:
            agent_white.replay(32)
        if agent_white.epsilon > agent_white.epsilon_min:
            agent_white.epsilon -= agent_white.epsilon_decay
        # Vẽ biểu đồ mỗi 50 game
        if game_index % 50 == 0 and game_index > 0:
            metrics.plot()
            print(f"Đã lưu biểu đồ tại training_metrics_{game_index}.png")

    # Lưu model và hiển thị kết quả
    torch.save(agent_white.model.state_dict(), MODEL_PATH)
    print("\n=== KẾT QUẢ SAU", num_games, "GAMES ===")
    print(f"WHITE thắng: {total_white_win}")
    print(f"BLACK thắng: {total_black_win}")
    print(f"Hòa: {total_draw}")