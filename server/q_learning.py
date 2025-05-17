import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

BLACK = 1
WHITE = 2

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TrainingMetrics:
    def __init__(self):
        self.win_rates = []
        self.losses = []
        self.epsilons = []
        self.rewards = []
        self.moving_avg_window = 20  # Cửa sổ trung bình động
        
    def update(self, win_rate, loss, epsilon, reward):
        self.win_rates.append(win_rate)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.rewards.append(reward)
        
    def plot(self, save_path="training_metrics.png"):
        plt.figure(figsize=(15, 10))
        
        # Tính toán trung bình động
        moving_avg = lambda x: np.convolve(x, np.ones(self.moving_avg_window)/self.moving_avg_window, mode='valid')
        
        # Vẽ 4 subplots
        plt.subplot(2, 2, 1)
        plt.plot(self.win_rates, label='Tỷ lệ thắng thực tế')
        plt.plot(moving_avg(self.win_rates), 
                label=f'Trung bình {self.moving_avg_window} game', 
                color='red', linewidth=2)
        plt.title('TỶ LỆ THẮNG CỦA AGENT')
        plt.xlabel('Số game')
        plt.ylabel('Tỷ lệ thắng')
        plt.ylim(0, 1)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(self.losses, alpha=0.3)
        plt.plot(moving_avg(self.losses), color='green', linewidth=2)
        #print(self.losses)
        plt.title('LOSS TRONG QUÁ TRÌNH TRAINING')
        plt.xlabel('Số batch training')
        plt.ylabel('Loss value')
        plt.yscale('log')  # Dùng scale log cho loss
        
        plt.subplot(2, 2, 3)
        plt.plot(self.epsilons)
        plt.title('GIÁ TRỊ EPSILON (EXPLORATION RATE)')
        plt.xlabel('Số game')
        plt.ylabel('Epsilon')
        plt.ylim(0, 1)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.rewards, alpha=0.3)
        plt.plot(moving_avg(self.rewards), color='purple', linewidth=2)
        plt.title('REWARD TRUNG BÌNH MỖI GAME')
        plt.xlabel('Số game')
        plt.ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),          # Dropout sau tầng 1
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),          # Dropout sau tầng 2
            nn.Linear(64, 64)           # Output Q-value cho 64 ô
        )
    def forward(self, x):
        return self.model(x)

class QNetworkAgent:
    def __init__(self):
        self.model = QNetwork()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.0001
        self.current_loss = 0.0
        self.losses = []

    def encode_state(self, obs):
        board = obs['board'].reshape(-1)
        player = obs['turn']
        if player == WHITE:
            board = np.where(board == BLACK, -1, board)
            board = np.where(board == WHITE, 1, board)
        return torch.tensor(board, dtype=torch.float32)

    def choose_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0)).squeeze(0)
            best_idx = max(valid_moves, key=lambda idx: q_values[idx].item())
            return best_idx

    def train(self, state, action, reward, next_state, done, next_valid_moves):
        self.model.train()

        state = state.unsqueeze(0)  # (1, 64)
        next_state = next_state.unsqueeze(0)

        with torch.no_grad():
            target_q = self.model(state).clone().squeeze(0)
            if done:
                target_q[action] = reward
            else:
                next_q = self.model(next_state).squeeze(0)
                if next_valid_moves:
                    max_q_next = max([next_q[i].item() for i in next_valid_moves])
                else:
                    max_q_next = 0.0
                target_q[action] = reward + self.gamma * max_q_next

        predicted_q = self.model(state).squeeze(0)
        loss = self.criterion(predicted_q, target_q)
        self.current_loss = loss.item()
        self.losses.append(self.current_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        