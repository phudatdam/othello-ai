import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

BLACK = 1
WHITE = 2

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),  # Giảm từ 256 xuống 128
            nn.ReLU(),
            #nn.BatchNorm1d(128),  # Thêm batch norm
            nn.Linear(128, 64),
            nn.ReLU(), 
            #nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.Tanh()  # Thêm tanh để bound output
        )


    def forward(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)



class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_valid_moves):
        self.buffer.append((state, action, reward, next_state, done, next_valid_moves))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
class QNetworkAgent:
    def __init__(self):
        self.model = QNetwork()
        self.target_model = QNetwork()  # Thêm target network
        self.target_model.load_state_dict(self.model.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.current_loss = 0.0
        self.losses = []
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 128
        self.max_grad_norm = 1.0

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
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0)).squeeze(0)
            best_idx = max(valid_moves, key=lambda idx: q_values[idx].item())
            return best_idx
        self.model.train()

    def train(self, state, action, reward, next_state, done, next_valid_moves):
        # Thêm kinh nghiệm mới vào replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done, next_valid_moves)

        # Chỉ train khi buffer đủ dữ liệu
        if len(self.replay_buffer) < self.batch_size:
            return
        self.model.train()

        # Lấy mẫu ngẫu nhiên từ buffer
        batch = self.replay_buffer.sample(self.batch_size)

        state_batch = torch.stack([s[0] for s in batch])
        action_batch = torch.tensor([s[1] for s in batch])
        reward_batch = torch.tensor([s[2] for s in batch], dtype=torch.float32)
        next_state_batch = torch.stack([s[3] for s in batch])
        done_batch = torch.tensor([s[4] for s in batch], dtype=torch.float32)
        next_valid_batch = [s[5] for s in batch]

        # Dự đoán Q hiện tại
        predicted_q = self.model(state_batch)  # (batch_size, 64)
        predicted_q = predicted_q.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Tính Q mục tiêu
        target_q = []
        with torch.no_grad():
            next_q_values = self.model(next_state_batch)
            for i in range(self.batch_size):
                if done_batch[i]:
                    target_q.append(reward_batch[i])
                else:
                    if next_valid_batch[i]:
                        max_next = max([next_q_values[i][j].item() for j in next_valid_batch[i]])
                    else:
                        max_next = 0.0
                    target_q.append(reward_batch[i] + self.gamma * max_next)

        target_q = torch.tensor(target_q, dtype=torch.float32)

        # Tính loss và cập nhật
        loss = self.criterion(predicted_q, target_q)
        self.current_loss = loss.item()
        self.losses.append(self.current_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if len(self.replay_buffer) % 100 == 0:  # Mỗi 100 steps
            self._update_target_network()

        # Giảm epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _update_target_network(self):
        # Soft update
        tau = 0.001
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)