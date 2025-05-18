import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from othello import Game
BOARD_SIZE = 64
WHITE = 2
BLACK = 1

class MinimaxQNetwork(nn.Module):
    def __init__(self):
        super(MinimaxQNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(66, 128),  # 64 trạng thái + 2 ô cho hành động và phản ứng đối thủ
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)  # Q-value
        )

    def forward(self, x):
        return self.model(x)

class MinimaxQAgent:
    def __init__(self):
        self.model = MinimaxQNetwork()
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

    def choose_action(self, observation, valid_agent_moves):
        best_action = None
        max_min_q = -float('inf')

        # Lấy đối tượng game từ observation hoặc môi trường
        current_game = self.get_game_from_observation(observation)

        # Mã hóa trạng thái 1 lần để dùng lại
        state_tensor = torch.tensor(self.encode_state(observation), dtype=torch.float32)

        for agent_action in valid_agent_moves:
            # Tạo bản sao game để mô phỏng
            temp_game = current_game.deepcopy()
            row, col = divmod(agent_action, 8)
            temp_game.play(temp_game.turn, row, col)

            # Lấy nước đi hợp lệ của đối thủ từ trạng thái mới
            opponent_moves = temp_game.get_valid_moves
            opponent_actions = [r * 8 + c for r, c in opponent_moves]

            if not opponent_actions:
                opponent_actions = [64]

            # Tính min Q-value cho hành động này
            min_q = float('inf')
            for opponent_action in opponent_actions:
                input_tensor = torch.cat([
                    state_tensor,
                    torch.tensor([agent_action, opponent_action], dtype=torch.float32)
                ])
                q_value = self.model(input_tensor.unsqueeze(0)).item()
                min_q = min(min_q, q_value)

        # Cập nhật hành động tốt nhất
        if min_q > max_min_q:
            max_min_q = min_q
            best_action = agent_action

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_agent_moves)

        return best_action

    def get_game_from_observation(self, obs):
        """Khôi phục trạng thái game từ observation."""
        game = Game()
        game.board_state = obs['board']
        game.turn = obs['turn']
        return game
    
    def train(self, state_tensor, action, opponent_action, reward, next_state_tensor, done, next_valid_moves, next_opponent_moves_dict):
        self.model.train()

        input_tensor = torch.cat([state_tensor, torch.tensor([action, opponent_action], dtype=torch.float32)]).unsqueeze(0)

        if done:
            target = torch.tensor([[reward]], dtype=torch.float32)
        else:
            future_qs = []
            for a in next_valid_moves:
                min_q = float('inf')
                for o in next_opponent_moves_dict:
                    next_input = torch.cat([next_state_tensor, torch.tensor([a, o], dtype=torch.float32)]).unsqueeze(0)
                    q_val = self.model(next_input).item()
                    min_q = min(min_q, q_val)
                future_qs.append(min_q)
            max_q = max(future_qs) if future_qs else 0.0
            target = torch.tensor([[reward + self.gamma * max_q]], dtype=torch.float32)

        pred = self.model(input_tensor)
        loss = self.criterion(pred, target)
        self.current_loss = loss.item()
        self.losses.append(self.current_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)