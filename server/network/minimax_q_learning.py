"""
Minimax Q-learning cho Othello (bản cải tiến).

Ý tưởng giữ nguyên như bản cũ: agent chọn nước a = argmax_a min_o Q(s, a, o).
Khác biệt:
- Thay vì đưa (s, a, o) dạng 4 tọa độ vào mạng, ta đánh giá thẳng bàn cờ SAU khi
  đi a và o: Q(s, a, o) = V(s_{a,o}). Input vẫn chỉ là 64 ô (mình = 1, địch = -1).
- Mạng CNN trên lưới 8x8 thay cho MLP.
- Toàn bộ lá (a, o) được đánh giá trong MỘT lần forward (batch), luật cờ dùng bitboard
  nên nhanh hơn rất nhiều so với deepcopy list.
- Target là lambda-return lấy từ giá trị tìm kiếm max-min ở nước sau (TD-leaf),
  nên reward cuối ván lan ngược về đầu ván nhanh hơn 1-step TD.
- Replay buffer + tăng dữ liệu bằng 8 phép đối xứng của bàn cờ.

V(board) luôn là giá trị theo góc nhìn của người SẮP ĐI, trong [-1, 1],
nên dữ liệu của cả quân Đen lẫn quân Trắng (self-play) dùng chung một mạng.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BLACK = 1
WHITE = 2

# ---------------------------------------------------------------------------
# Bitboard: bit thứ r*8+c là ô (r, c)
# ---------------------------------------------------------------------------
FULL = (1 << 64) - 1
NOT_A = 0xFEFEFEFEFEFEFEFE  # bỏ cột 0
NOT_H = 0x7F7F7F7F7F7F7F7F  # bỏ cột 7

_SHIFTS = (
    lambda b: (b << 1) & NOT_A,          # phải
    lambda b: (b >> 1) & NOT_H,          # trái
    lambda b: (b << 8) & FULL,           # xuống
    lambda b: b >> 8,                    # lên
    lambda b: (b << 9) & NOT_A & FULL,   # xuống-phải
    lambda b: (b << 7) & NOT_H & FULL,   # xuống-trái
    lambda b: (b >> 7) & NOT_A,          # lên-phải
    lambda b: (b >> 9) & NOT_H,          # lên-trái
)


def legal_moves(p, o):
    """Bitmask các nước hợp lệ của người có quân p, đối thủ có quân o."""
    empty = ~(p | o) & FULL
    moves = 0
    for sh in _SHIFTS:
        x = sh(p) & o
        for _ in range(5):
            x |= sh(x) & o
        moves |= sh(x) & empty
    return moves


def apply_move(p, o, sq):
    """Đi ô sq (0..63). Trả về (p_mới, o_mới)."""
    m = 1 << sq
    flips = 0
    for sh in _SHIFTS:
        f = 0
        x = sh(m)
        while x & o:
            f |= x
            x = sh(x)
        if x & p:
            flips |= f
    return p | m | flips, o & ~flips


def bits(mask):
    """Liệt kê chỉ số các bit bật."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def from_board(board_state, player):
    """List 8x8 (0/1/2) -> (quân của player, quân đối thủ) dạng bitboard."""
    p = o = 0
    for r in range(8):
        for c in range(8):
            v = board_state[r][c]
            if v == player:
                p |= 1 << (r * 8 + c)
            elif v != 0:
                o |= 1 << (r * 8 + c)
    return p, o


def to_board(p, o, player):
    """Bitboard -> list 8x8 (0/1/2) để dùng với minimax/evaluator cũ."""
    opp = WHITE if player == BLACK else BLACK
    board = [[0] * 8 for _ in range(8)]
    for sq in bits(p):
        board[sq // 8][sq % 8] = player
    for sq in bits(o):
        board[sq // 8][sq % 8] = opp
    return board


_IDX = np.arange(64, dtype=np.uint64)


def encode(p, o):
    """64 ô theo góc nhìn người có quân p: mình = 1, địch = -1, trống = 0."""
    pb = (np.uint64(p) >> _IDX) & np.uint64(1)
    ob = (np.uint64(o) >> _IDX) & np.uint64(1)
    return pb.astype(np.float32) - ob.astype(np.float32)


def final_value(p, o):
    """Kết quả ván theo góc nhìn p: thắng/thua là chính, cộng một chút theo chênh lệch quân."""
    diff = bin(p).count("1") - bin(o).count("1")
    return float(np.sign(diff)) * 0.8 + diff / 64.0 * 0.2


# ---------------------------------------------------------------------------
# Mạng giá trị
# ---------------------------------------------------------------------------
class MinimaxQNetwork(nn.Module):
    """CNN trên lưới 8x8, input vẫn là 64 ô."""

    def __init__(self, channels=32, num_blocks=4):
        super().__init__()
        self.inp = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)
        x = F.relu(self.inp(x))
        for conv in self.blocks:
            x = F.relu(x + conv(x))  # residual
        return self.head(x).squeeze(-1)


def augment(x):
    """Chia batch làm 8 phần, mỗi phần áp một trong 8 phép đối xứng của bàn cờ."""
    x = x.view(-1, 8, 8)
    out = []
    for i, g in enumerate(torch.chunk(x, 8)):
        if i & 4:
            g = torch.flip(g, dims=(2,))
        out.append(torch.rot90(g, k=i & 3, dims=(1, 2)))
    return torch.cat(out).reshape(-1, 64)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class MinimaxQAgent:
    def __init__(self, lr=1e-3, buffer_size=200_000, lam=0.7, device="cpu"):
        self.device = device
        self.model = MinimaxQNetwork().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = deque(maxlen=buffer_size)
        self.lam = lam
        self.epsilon = 0.3
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998  # giảm theo ván, không phải theo sample
        self.current_loss = 0.0
        self.losses = []

    # ---- đánh giá -------------------------------------------------------
    @torch.no_grad()
    def _values(self, model, states):
        if not states:
            return np.zeros(0, dtype=np.float32)
        x = torch.from_numpy(np.stack(states)).to(self.device)
        return model(x).cpu().numpy()

    def search(self, p, o, model=None):
        """
        max_a min_o V(s_{a,o}) cho người có quân p (đang tới lượt).
        Trả về (danh sách ô hợp lệ, giá trị max-min của từng ô).
        """
        model = model or self.model
        moves = bits(legal_moves(p, o))
        leaves = []        # trạng thái cần đưa qua mạng
        leaf_sign = []     # +1: p sắp đi, -1: đối thủ sắp đi
        children = []      # với mỗi a: list chỉ số lá, hoặc ("v", giá trị cố định)

        def add_leaf(pp, oo, sign):
            leaves.append(encode(pp, oo))
            leaf_sign.append(sign)
            return len(leaves) - 1

        for a in moves:
            p1, o1 = apply_move(p, o, a)
            replies = bits(legal_moves(o1, p1))
            entries = []
            if not replies:
                # đối thủ mất lượt
                if legal_moves(p1, o1):
                    entries.append(add_leaf(p1, o1, 1.0))
                else:
                    entries.append(("v", final_value(p1, o1)))
            else:
                for r in replies:
                    o2, p2 = apply_move(o1, p1, r)
                    if legal_moves(p2, o2):
                        entries.append(add_leaf(p2, o2, 1.0))
                    elif legal_moves(o2, p2):
                        # mình mất lượt, đối thủ đi tiếp
                        entries.append(add_leaf(o2, p2, -1.0))
                    else:
                        entries.append(("v", final_value(p2, o2)))
            children.append(entries)

        vals = self._values(model, leaves) * np.array(leaf_sign, dtype=np.float32)
        scores = [min(e[1] if isinstance(e, tuple) else float(vals[e]) for e in entries)
                  for entries in children]
        return moves, scores

    def choose(self, p, o, greedy=False, model=None):
        """Trả về (ô được chọn, giá trị max-min của trạng thái)."""
        moves, scores = self.search(p, o, model)
        best = int(np.argmax(scores))
        value = scores[best]
        if not greedy and random.random() < self.epsilon:
            return random.choice(moves), value
        return moves[best], value

    # ---- API dùng với Game (board_state dạng list) -----------------------
    def choose_action(self, board_state, player, greedy=True):
        p, o = from_board(board_state, player)
        if not legal_moves(p, o):
            return None
        sq, _ = self.choose(p, o, greedy=greedy)
        return divmod(sq, 8)

    # ---- học -------------------------------------------------------------
    def store_game(self, states, search_values, final_z):
        """
        states[t]: trạng thái lúc agent tới lượt (encode theo góc nhìn agent)
        search_values[t]: giá trị max-min tính được tại states[t]
        final_z: kết quả cuối ván theo góc nhìn agent
        Target lambda-return: G_t = (1-lam) * search_values[t+1] + lam * G_{t+1}
        """
        g = final_z
        for t in reversed(range(len(states))):
            if t + 1 < len(states):
                g = (1 - self.lam) * search_values[t + 1] + self.lam * g
            self.memory.append((states[t], g))

    def replay(self, batch_size=256):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        x = torch.from_numpy(np.stack([s for s, _ in batch])).to(self.device)
        y = torch.tensor([g for _, g in batch], dtype=torch.float32, device=self.device)
        pred = self.model(augment(x))
        loss = F.smooth_l1_loss(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.current_loss = loss.item()
        self.losses.append(self.current_loss)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
