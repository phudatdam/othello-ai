import random
from collections import defaultdict
from test_enviroment import OthelloEnv  # Import môi trường Othello của bạn
from othello import BLACK, WHITE
import pickle

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q = defaultdict(lambda: defaultdict(float))  # Q[state][action]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_id(self, obs):
        # Chuyển trạng thái về dạng hashable (dùng tuple)
        board = tuple(obs['board'].flatten())
        turn = obs['turn']
        return (board, turn)

    def choose_action(self, obs, valid_actions):
        state = self.get_state_id(obs)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_values = self.Q[state]
        return max(valid_actions, key=lambda a: q_values[a])

    def learn(self, obs, action, reward, next_obs, done, valid_next_actions):
        state = self.get_state_id(obs)
        next_state = self.get_state_id(next_obs)
        q_sa = self.Q[state][action]
        
        if done or not valid_next_actions:
            q_target = reward
        else:
            q_target = reward + self.gamma * max([self.Q[next_state][a] for a in valid_next_actions])

        self.Q[state][action] += self.alpha * (q_target - q_sa)

if __name__ == "__main__":
    env = OthelloEnv()
    agent = QLearningAgent()

    num_episodes = 10000

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            # Lấy danh sách hành động hợp lệ
            valid_moves = env.game.get_valid_moves
            valid_actions = [row * env.board_size + col for row, col in valid_moves]
            
            if not valid_actions:
                # Nếu không còn hành động nào hợp lệ thì dừng
                break

            action = agent.choose_action(obs, valid_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            valid_moves = env.game.get_valid_moves
            next_valid_actions = [row * env.board_size + col for row, col in valid_moves]

            agent.learn(obs, action, reward, next_obs, done, next_valid_actions)

            obs = next_obs

        if (episode + 1) % 500 == 0:
            print(f"Tập {episode + 1} hoàn tất.")

    env.close()

     # Lưu Q-table ra file
    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)