import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from othello import Game # Giả sử bạn có file othello.py và utils.py

BLACK = 1
WHITE = 2
EMPTY = 0

class PolicyNetwork(nn.Module):
    """
    Mạng chính sách cho Othello.
    Đầu vào: Trạng thái bàn cờ (8x8 = 64 phần tử).
    Đầu ra: Phân phối xác suất cho 64 nước đi có thể (8x8 ô).
    """
    def __init__(self, input_dim=64, output_dim=64, hidden_dim=256, num_layers=3):
        super(PolicyNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU()) # Thêm ReLU sau lớp đầu tiên

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        # Sử dụng Softmax để đảm bảo đầu ra là phân phối xác suất
        # cho các nước đi hợp lệ.
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PolicyAgent:
    """
    Agent sử dụng mạng chính sách để chọn nước đi và học hỏi.
    """
    def __init__(self, learning_rate=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.policy_network = PolicyNetwork()
        # Sử dụng Adam optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        # CrossEntropyLoss phù hợp cho bài toán phân loại (chọn nước đi)
        self.criterion = nn.CrossEntropyLoss()
        self.gamma = gamma # Hệ số chiết khấu cho reward
        self.epsilon = epsilon_start # Tỷ lệ khám phá (exploration rate)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000) # Replay buffer để lưu trữ kinh nghiệm
        self.losses = [] # Lưu trữ loss để theo dõi quá trình huấn luyện
        self.current_loss = 0
        self.train_step = 0

    def encode_state(self, obs):
        """
        Mã hóa trạng thái bàn cờ thành vector đầu vào cho mạng nơ-ron.
        - Chuyển đổi bàn cờ 8x8 thành vector 1D 64 phần tử.
        - Chuẩn hóa giá trị: quân của agent là 1, quân đối thủ là -1, ô trống là 0.
        """
        board_raw = obs['board'] # Lấy dữ liệu board thô
        
        # Đảm bảo board_raw là một numpy array và làm phẳng nó
        if isinstance(board_raw, torch.Tensor):
            board_np = board_raw.cpu().numpy().flatten() # Chuyển Tensor sang NumPy và làm phẳng
        elif isinstance(board_raw, np.ndarray):
            board_np = board_raw.flatten() # Nếu đã là NumPy array, chỉ cần làm phẳng
        else: # Giả định là list of lists hoặc định dạng khác có thể chuyển đổi thành NumPy array
            board_np = np.array(board_raw).flatten()
        
        board = board_np # board bây giờ là một numpy array 1D
        
        player = obs['turn']

        # Chuẩn hóa board state theo góc nhìn của người chơi hiện tại
        if player == WHITE:
            # Nếu là quân trắng, quân trắng là 1, quân đen là -1
            board_encoded = np.where(board == WHITE, 1, np.where(board == BLACK, -1, 0))
        else:
            # Nếu là quân đen, quân đen là 1, quân trắng là -1
            board_encoded = np.where(board == BLACK, 1, np.where(board == WHITE, -1, 0))
        
        return torch.tensor(board_encoded, dtype=torch.float32)

    def choose_action(self, obs, valid_moves):
        """
        Chọn một nước đi dựa trên chính sách hiện tại (policy network)
        hoặc khám phá ngẫu nhiên (epsilon-greedy).
        """
        if random.random() < self.epsilon:
            # Khám phá: chọn một nước đi ngẫu nhiên trong các nước đi hợp lệ
            return random.choice(valid_moves)
        
        self.policy_network.eval() # Chuyển sang chế độ đánh giá
        state_tensor = self.encode_state(obs)
        
        with torch.no_grad():
            # Lấy log_probabilities từ mạng chính sách
            # Chúng ta sẽ áp dụng softmax sau để có phân phối xác suất
            action_logits = self.policy_network(state_tensor.unsqueeze(0)).squeeze(0)
            
            # Tạo một mask cho các nước đi không hợp lệ
            # Đặt giá trị rất nhỏ cho các nước đi không hợp lệ để xác suất của chúng gần bằng 0
            mask = torch.full_like(action_logits, float('-inf'))
            for r, c in valid_moves:
                idx = r * 8 + c # Chuyển đổi (row, col) thành chỉ số 1D
                mask[idx] = 0 # Đặt 0 cho các nước đi hợp lệ để không bị ảnh hưởng bởi -inf

            # Áp dụng mask và sau đó Softmax để có phân phối xác suất chỉ trên các nước đi hợp lệ
            masked_action_logits = action_logits + mask
            action_probs = torch.softmax(masked_action_logits, dim=-1)

            # Chọn nước đi dựa trên phân phối xác suất
            # Chuyển đổi phân phối xác suất thành numpy array để sử dụng np.random.choice
            action_indices = np.arange(action_probs.size(0))
            chosen_idx = np.random.choice(action_indices, p=action_probs.cpu().numpy())
            
            # Chuyển đổi chỉ số 1D trở lại (row, col)
            chosen_row = chosen_idx // 8
            chosen_col = chosen_idx % 8
            
        self.policy_network.train() # Chuyển về chế độ huấn luyện
        return (chosen_row, chosen_col)

    def store_experience(self, state, action, reward, next_state, done): #replay_buffer_save
        """
        Lưu trữ kinh nghiệm vào replay buffer.
        State và next_state đã được mã hóa thành tensor.
        Action là tuple (row, col).
        """
        # Chuyển đổi action tuple thành chỉ số 1D
        # Kiểm tra nếu action là PASS_ACTION_VALUE (hoặc bất kỳ số nguyên nào khác)
        if isinstance(action, int):
            # Nếu là hành động PASS, chúng ta có thể lưu trữ nó với một chỉ số đặc biệt
            # hoặc bỏ qua không lưu vào replay buffer nếu bạn không muốn học từ các lượt pass.
            # Ở đây, tôi sẽ lưu trữ nó với PASS_ACTION_VALUE.
            action_idx = action 
        else:
            # Nếu action là tuple (row, col), chuyển đổi thành chỉ số 1D
            action_idx = action[0] * 8 + action[1]
        self.memory.append((state, action_idx, reward, next_state, done))

    def update_policy(self, batch_size):#replay
        """
        Huấn luyện mạng chính sách bằng cách lấy mẫu từ replay buffer.
        Sử dụng thuật toán REINFORCE hoặc tương tự.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.stack([s for s, _, _, _, _ in minibatch])
        actions_indices = torch.tensor([a for _, a, _, _, _ in minibatch], dtype=torch.long)
        rewards = torch.tensor([r for _, _, r, _, _ in minibatch], dtype=torch.float32)
        next_states = torch.stack([ns for _, _, _, ns, _ in minibatch])
        dones = torch.tensor([d for _, _, _, _, d in minibatch], dtype=torch.float32)

        # Tính toán log xác suất của các hành động đã chọn
        # policy_network trả về logits, sau đó chúng ta lấy log_softmax
        action_logits = self.policy_network(states)
        log_probs = torch.log_softmax(action_logits, dim=-1)
        
        # Lấy log_prob của hành động đã thực hiện
        chosen_action_log_probs = log_probs.gather(1, actions_indices.unsqueeze(1)).squeeze(1)

        # Tính toán Gt (Discounted Return)
        # Đây là một cách đơn giản hóa, trong thực tế cần tính toán kỹ hơn
        # Ví dụ: Gt = reward + gamma * V(next_state)
        # Để đơn giản, chúng ta sẽ dùng reward trực tiếp cho mỗi bước huấn luyện
        # Hoặc có thể tính Monte Carlo return cho toàn bộ episode
        # Tuy nhiên, với replay buffer, chúng ta thường dùng TD-error.
        # Đối với Policy Gradient, chúng ta cần một baseline hoặc Monte Carlo return.
        
        # Để đơn giản cho ví dụ này, chúng ta sẽ sử dụng reward như là "advantage"
        # Một cách tiếp cận phổ biến hơn là dùng Value Network để ước lượng baseline.
        # Hoặc tính toán Monte Carlo return cho toàn bộ episode rồi mới update.
        # Ở đây, chúng ta sẽ giả định reward là một dạng của advantage.
        
        # Loss cho Policy Gradient: -log_prob * advantage
        # Nếu reward là dương, chúng ta muốn tăng log_prob của hành động đó.
        # Nếu reward là âm, chúng ta muốn giảm log_prob của hành động đó.
        loss = - (chosen_action_log_probs * rewards).mean() # Nhân với rewards để làm advantage

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient để tránh gradient exploding
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        self.current_loss=loss.item()
        self.losses.append(loss.item())
        self.train_step += 1

        # Giảm epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
'''
if __name__ == '__main__':
    from othello import Game
    # Để chạy ví dụ, bạn cần có file utils.py và Game class trong othello.py
    # Đảm bảo Game.get_valid_moves trả về list các tuple (row, col)

    # Ví dụ về cách sử dụng PolicyAgent
    agent = PolicyAgent()
    game = Game()

    # Tạo một trạng thái giả định (ví dụ: trạng thái ban đầu của Othello)
    initial_obs = {
        'board': game.board_state,
        'turn': game.turn
    }
    
    print("Trạng thái ban đầu:")
    for row in initial_obs['board']:
        print(row)
    print(f"Lượt của: {'BLACK' if initial_obs['turn'] == BLACK else 'WHITE'}")

    # Lấy các nước đi hợp lệ
    valid_moves = game.get_valid_moves
    print(f"Các nước đi hợp lệ: {valid_moves}")

    if valid_moves:
        # Chọn một nước đi
        chosen_action = agent.choose_action(initial_obs, valid_moves)
        print(f"Agent chọn nước đi: {chosen_action}")

        # Giả lập một bước chơi
        # Để đơn giản, chúng ta sẽ không thực sự chơi game ở đây,
        # chỉ mô phỏng việc lưu trữ kinh nghiệm và cập nhật chính sách.
        reward = 1.0 if random.random() > 0.5 else -1.0 # Reward ngẫu nhiên
        next_state_obs = initial_obs # Giả sử trạng thái không đổi để đơn giản
        done = False # Giả sử game chưa kết thúc

        # Mã hóa trạng thái và lưu trữ kinh nghiệm
        encoded_state = agent.encode_state(initial_obs)
        encoded_next_state = agent.encode_state(next_state_obs)
        agent.store_experience(encoded_state, chosen_action, reward, encoded_next_state, done)

        # Cập nhật chính sách
        print("Cập nhật chính sách...")
        for _ in range(10): # Thực hiện vài lần cập nhật để thấy loss thay đổi
            agent.update_policy(batch_size=1) # Batch size nhỏ để dễ thấy
        print(f"Loss sau khi cập nhật: {agent.losses[-1]}")
    else:
        print("Không có nước đi hợp lệ trong trạng thái ban đầu.")

    # Ví dụ về cách lấy phân phối xác suất từ mạng chính sách (sau khi huấn luyện)
    agent.policy_network.eval()
    state_tensor_eval = agent.encode_state(initial_obs)
    with torch.no_grad():
        action_logits_eval = agent.policy_network(state_tensor_eval.unsqueeze(0)).squeeze(0)
        
        mask_eval = torch.full_like(action_logits_eval, float('-inf'))
        for r, c in valid_moves:
            idx = r * 8 + c
            mask_eval[idx] = 0
        
        masked_action_logits_eval = action_logits_eval + mask_eval
        final_action_probs = torch.softmax(masked_action_logits_eval, dim=-1)
        
        print("\nPhân phối xác suất nước đi sau khi huấn luyện (chỉ các nước đi hợp lệ):")
        for r, c in valid_moves:
            idx = r * 8 + c
            print(f"({r}, {c}): {final_action_probs[idx].item():.4f}")

'''

