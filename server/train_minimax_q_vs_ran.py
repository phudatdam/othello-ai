import os
import torch
import numpy as np
from test_environment import OthelloEnv
from ai.ai_player import MinimaxPlayer, RandomPlayer
from network.minimax_q_learning import MinimaxQAgent
from network.metrics import TrainingMetrics
import utils
CORNERS = [(0,0), (0,7), (7,0), (7,7)]
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
        print("Model loaded from", MODEL_PATH)
    else:
        print("No saved model found, training from scratch.")
    game_total = 0
    for prob in range (1, 2):
        #minimax_agent_black = MinimaxPlayer(BLACK, 2, 0.5)
        #agent_white.epsilon = 1.0
        for game_index_per_depth in range(num_games):
            game_total += 1
            #print(f"\n=== GAME {game_total} ===")
            
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
            white_reward = 0
            black_reward = 0
            while not done:
                # === PHASE 1: BLACK'S TURN ===

                move = random_agent_black.play(env.game)
                #temp = (temp+1)%10
                if move is not None:
                    row, col = move
                else:
                    row, col = -1, -1  # Pass
                    env.game.turn = WHITE
                action_flat = row * env.board_size + col
                next_observation, black_reward, terminated, truncated, info = env.step(action_flat)

                done = terminated or truncated
                if prev_state is not None:
                    next_state = agent_white.encode_state(next_observation)
                    temp_reward = black_reward + white_reward
                    #Thưởng thêm nếu lấy được góc
                    if (white_action in CORNERS):
                        temp_reward+=0.5
                    if ((row, col) in CORNERS):
                        temp_reward-=0.5
                    agent_white.replay_buffer_save(
                        state=prev_state,
                        action=white_action,
                        opponent_action=(row, col),
                        reward=temp_reward,
                        next_state=next_state,
                        done=done
                    )
                total_reward += black_reward
                
                observation = next_observation
                #env.render()
                if done:
                    break
                # === PHASE 2: WHITE'S TURN ===
                valid_moves_white = env.game.get_valid_moves

                prev_state = agent_white.encode_state(observation)
                if valid_moves_white:
                    action = agent_white.choose_action(observation, valid_moves_white)
                    row, col = action
                else:
                    env.game.turn = BLACK
                    row, col = -1, -1  # Pass
                action_flat = row * env.board_size + col
                next_observation, white_reward, terminated, truncated, info = env.step(action_flat)
                white_action = (row, col)
                done = terminated or truncated
                observation = next_observation
                total_reward += white_reward
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
            if agent_white.epsilon > agent_white.epsilon_min:
                agent_white.epsilon *= agent_white.epsilon_decay
            metrics.update(
                win_label=1 if winner == WHITE else 0,
                loss=agent_white.current_loss,
                epsilon=agent_white.epsilon,
                reward=total_reward
            )
            
            # Train định kỳ
            if len(agent_white.memory) > 64:
                agent_white.replay(64)
            
            # Vẽ biểu đồ mỗi 50 game
            if game_total % 20 == 0 and game_total > 0:
                metrics.plot()
                print(f"Đã lưu biểu đồ tại training_metrics_{game_total}.png")
                print("\n=== KẾT QUẢ SAU", num_games, "GAMES ===")
                print(f"WHITE thắng: {total_white_win}")
                print(f"BLACK thắng: {total_black_win}")
                print(f"Hòa: {total_draw}")
                torch.save(agent_white.model.state_dict(), MODEL_PATH)