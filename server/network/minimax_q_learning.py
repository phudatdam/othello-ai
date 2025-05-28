import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from othello import Game
from collections import deque
import copy
import utils
BLACK = 1
WHITE = 2

""" Khởi tạo mạng Q-Learning"""
class MinimaxQNetwork(nn.Module):
    def __init__(self, input_dim=68, output_dim=1, hidden_dim=128, num_layers=4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim),)

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))

        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)

class MinimaxQAgent:
    def __init__(self):
        self.model = MinimaxQNetwork()
        self.target_model = MinimaxQNetwork()
        self.target_model.load_state_dict(self.model.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0
        self.epsilon_decay = 0.995
        self.current_loss = 0.0
        self.losses = []
        self.memory = deque(maxlen=15000)
        self.update_target_steps = 1000
        self.train_step = 0
        self.max_grad_norm = 0.1

    def encode_state(self, obs):
        board = obs['board'].reshape(-1)
        player = obs['turn']
        if player == WHITE:
            board = np.where(board == BLACK, -1, board)
            board = np.where(board == WHITE, 1, board)
        else:
            board = np.where(board == BLACK, 1, board)
            board = np.where(board == WHITE, -1, board)
        return torch.tensor(board, dtype=torch.float32)

    

    def choose_exploitation_action(self, obs, valid_agent_moves):
        self.model.eval()
        state_tensor = self.encode_state(obs)
        best_action = None
        best_value = -float('inf')

        # Tạo bản sao game từ observation
        current_board = np.array(obs['board']).reshape(8, 8).tolist()
        current_turn = obs['turn']

        for a in valid_agent_moves:
            # Tạo bản sao sâu để mô phỏng
            temp_game = Game()
            temp_game.board_state = copy.deepcopy(current_board)
            temp_game.turn = current_turn

            # Thực hiện action của agent
            try:
                temp_game.play(temp_game.turn, a[0], a[1])
            except ValueError:
                continue  # Bỏ qua action không hợp lệ

            # Lấy nước đi hợp lệ của đối thủ
            valid_opponent_moves = temp_game.get_valid_moves
            if not valid_opponent_moves:
                self.model.train()
                return a  # Đối thủ không có nước đi

            # Tính min Q-value cho từng phản ứng của đối thủ
            min_q = 1.0
            for o in valid_opponent_moves:
                # Tạo input tensor: state + (a, o)
                input_tensor = torch.cat([
                    state_tensor,
                    torch.tensor([a[0]/7.0, a[1]/7.0, o[0]/7.0, o[1]/7.0], dtype=torch.float32)
                ]).unsqueeze(0)

                with torch.no_grad():
                    q_val = self.model(input_tensor).item()

                min_q = min(min_q, q_val)

            if min_q > best_value:
                best_value = min_q
                best_action = a

        self.model.train()
        return best_action

    def choose_action(self, obs, valid_agent_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_agent_moves)
        return self.choose_exploitation_action(obs, valid_agent_moves)

    def replay_buffer_save(self, state, action, opponent_action, reward, next_state, done):
        self.memory.append((state, action, opponent_action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, opponent_action, reward, next_state, done in minibatch:
            state_tensor = state
            next_state_tensor = next_state

            input_tensor = torch.cat([
                state_tensor,
                torch.tensor([action[0]/7.0, action[1]/7.0, opponent_action[0]/7.0, opponent_action[1]/7.0], dtype=torch.float32)
            ]).unsqueeze(0)

            if done:
                target = torch.tensor([[reward]], dtype=torch.float32)
            else:
                board_arr = next_state_tensor.detach().cpu().numpy().reshape(8, 8)
                converted_board = np.where(board_arr == 1, 2, board_arr)
                converted_board = np.where(converted_board == -1, 1, converted_board)
                temp_board_state = converted_board.astype(int).tolist()
                
                valid_agent_moves = utils.get_valid_moves(temp_board_state, WHITE)

                if valid_agent_moves:
                    max_min_q = -1.0
                    for a_prime in valid_agent_moves:
                        #Nếu a_prime là góc
                        #if (a_prime[0], a_prime[1]) in [(0,0), (0,7), (7,0), (7,7)]:
                        #   max_min_q = 1.5
                        #   print("You make a corner so claim a reward")
                        #   break

                        temp_game_copy = Game()
                        temp_game_copy.board_state = copy.deepcopy(temp_board_state)
                        temp_game_copy.turn = WHITE
                        try:
                            temp_game_copy.play(WHITE, a_prime[0], a_prime[1])
                        except:
                            continue
                        
                        
                        valid_opponent_moves = temp_game_copy.get_valid_moves

                        if not valid_opponent_moves:
                            min_q = 1.2
                            #print("You make opponent pass so claim a reward")
                        else:
                            min_q = 1.0
                            for o_prime in valid_opponent_moves:
                                next_input = torch.cat([
                                    next_state_tensor,
                                    torch.tensor([a_prime[0]/7.0, a_prime[1]/7.0, o_prime[0]/7.0, o_prime[1]/7.0], dtype=torch.float32)
                                ]).unsqueeze(0)
                                with torch.no_grad():
                                    q_val = self.target_model(next_input).item()
                                min_q = min(min_q, q_val)
        

                        max_min_q = max(max_min_q, min_q)

                else:
                    # WHITE mất lượt → chỉ duyệt nước đi của BLACK
                    valid_opponent_moves = utils.get_valid_moves(temp_board_state, BLACK)
                    if not valid_opponent_moves:
                        min_q = 0.0
                    else:
                        #print("You already don't have turn so claim a punishment")
                        min_q = -0.2
                        for o_prime in valid_opponent_moves:
                            next_input = torch.cat([
                                next_state_tensor,
                                torch.tensor([-1.0/7.0, -1.0/7.0, o_prime[0]/7.0, o_prime[1]/7.0], dtype=torch.float32)
                            ]).unsqueeze(0)
                            with torch.no_grad():
                                q_val = self.target_model(next_input).item()
                            min_q = min(min_q, q_val)
                    max_min_q = min_q

                target = torch.tensor([[max(-1.0, min(1.0, reward + self.gamma * max_min_q))]], dtype=torch.float32)


            pred = self.model(input_tensor)
            loss = self.criterion(pred, target)
            self.current_loss = loss.item()
            self.losses.append(self.current_loss)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.train_step += 1
            if self.train_step % self.update_target_steps == 0:
                self.target_model.load_state_dict(self.model.state_dict())