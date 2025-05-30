from test_environment import OthelloEnv  # Môi trường Othello
from othello import BLACK, WHITE
from network.q_learning import QNetworkAgent  # Agent dùng mạng nơ-ron
from network.metrics import TrainingMetrics
from ai.ai_player import RandomPlayer, MinimaxPlayer  # Random player
import os
import torch
BATCH_SIZE=32
MODEL_PATH = "models/q_network.pt"
PASS_ACTION = -9
if __name__ == "__main__":
    metrics = TrainingMetrics()
    total_black_win = 0
    total_white_win = 0
    total_draw = 0

    num_games = 1000

    # Tải model đã huấn luyện
    agent_white = QNetworkAgent()
    if os.path.exists(MODEL_PATH):
        agent_white.model.load_state_dict(torch.load(MODEL_PATH))
        agent_white.model.eval()  # Đặt model về chế độ đánh giá
        print("Model loaded from", MODEL_PATH)
    else:
        print("No saved model found, training from scratch.")
    minimax_agent_black = MinimaxPlayer(BLACK, 2, 0.5)
    game_total = 0
    for game_index in range(num_games):
        game_total += 1
        env = OthelloEnv()
        observation, info = env.reset()
        done = False
        current_player = BLACK
        #random_agent_black = RandomPlayer(env.game)
        total_reward = 0

        while not done:
            if current_player == WHITE:
                state = agent_white.encode_state(observation)
                valid_moves = [r * 8 + c for r, c in env.game.get_valid_moves]
                if valid_moves:
                    action = agent_white.choose_action(state, valid_moves)
                else:
                    action = PASS_ACTION  # Pass (nếu bạn định nghĩa pass là action -9)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = agent_white.encode_state(next_observation)
                agent_white.replay_buffer_save(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                # Train ngay sau lượt của WHITE
            else:
                move = minimax_agent_black.play(env.game)
                if move is not None:
                    row, col = move
                    action = row * env.board_size + col
                else:
                    action = PASS_ACTION  # Pass
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            observation = next_observation
            current_player = 3 - current_player
            
        total_reward += reward
            #env.render()

        # Sau khi kết thúc game
        
        winner = env.game.get_winner()
        
        if winner == 2:
            total_white_win += 1
        elif winner == 1:
            total_black_win += 1
        else:
            total_draw += 1
    
        # Sau mỗi game
        if len(agent_white.memory) > BATCH_SIZE:
                    agent_white.replay(BATCH_SIZE)
        
        metrics.update(
            win_label=1 if winner == WHITE else 0,
            loss=agent_white.current_loss,
            epsilon=agent_white.epsilon,
            reward=total_reward
        )
        if agent_white.epsilon > agent_white.epsilon_min:
                agent_white.epsilon *= agent_white.epsilon_decay
        if len(agent_white.memory) > BATCH_SIZE:
                agent_white.replay(BATCH_SIZE)
        # Vẽ biểu đồ mỗi 50 game
        if game_total % 50 == 0 and game_total > 0:
            metrics.plot()
            """print(f"Đã lưu biểu đồ tại training_metrics_{game_total}.png")
            print("\n=== KẾT QUẢ SAU", num_games, "GAMES ===")
            print(f"WHITE thắng: {total_white_win}")
            print(f"BLACK thắng: {total_black_win}")
            print(f"Hòa: {total_draw}")"""
            torch.save(agent_white.model.state_dict(), MODEL_PATH)
        # Lưu model và hiển thị kết quả
            
    # Vẽ biểu đồ cuối cùng
    metrics.plot("final_training_metrics.png")
    print(f"Saved model to {MODEL_PATH}")