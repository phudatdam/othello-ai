"""
Train Minimax Q-learning agent.

- Curriculum: đối thủ dễ trước (random), sau đó minimax pha ngẫu nhiên, rồi minimax
  độ sâu tăng dần. Chỉ lên level khi agent thắng ổn định ở level hiện tại.
- Self-play: một phần số ván đấu với bản sao cũ của chính mình.
- Agent chơi luân phiên cả quân Đen và quân Trắng.

Chạy:  python train_minimax_q_vs_ran.py --games 3000
       python train_minimax_q_vs_ran.py --games 3000 --resume
       python train_minimax_q_vs_ran.py --games 3000 --resume --level 6 --epsilon 0.1   (train tiếp ở độ sâu 4)
       python train_minimax_q_vs_ran.py --eval 40      (chỉ đánh giá model đã lưu)
"""
import argparse
import copy
import os
import random
import time
from collections import deque

import numpy as np
import torch

from ai import minimax
from network import fast_minimax
from network.metrics import TrainingMetrics
from network.minimax_q_learning import (
    BLACK, WHITE, MinimaxQAgent, apply_move, bits, encode, final_value, legal_moves, to_board,
)

MODEL_PATH = "models/minimax_q_cnn.pt"

# (loại, độ sâu, xác suất đi ngẫu nhiên)
# "fast": minimax bitboard (network/fast_minimax.py), dùng khi train vì nhanh.
# "minimax": minimax gốc của project (ai/minimax.py), chỉ dùng khi đánh giá vì rất chậm.
LEVELS = [
    ("random", 0, 1.0),
    ("fast", 1, 0.5),
    ("fast", 1, 0.0),
    ("fast", 2, 0.3),
    ("fast", 2, 0.0),
    ("fast", 3, 0.0),
    ("fast", 4, 0.0),
]
OPENING_RANDOM_PLIES = 4  # khi đánh giá, 4 nước đầu đi ngẫu nhiên để các ván khác nhau
PROMOTE_WIN_RATE = 0.65
PROMOTE_WINDOW = 50
SELF_PLAY_RATIO = 0.3
SNAPSHOT_EVERY = 200
TRAIN_STEPS_PER_GAME = 8

# Thế cờ ban đầu dạng bitboard (bit r*8+c): Đen ở (3,4),(4,3); Trắng ở (3,3),(4,4)
START = {BLACK: (1 << 28) | (1 << 35), WHITE: (1 << 27) | (1 << 36)}


def opponent_move(p, o, color, kind, depth, p_random):
    moves = bits(legal_moves(p, o))
    if random.random() < p_random:
        return random.choice(moves)
    if kind == "fast":
        return fast_minimax.best_move(p, o, depth)
    if len(minimax.transposition_table) > 2_000_000:
        minimax.transposition_table.clear()
    r, c = minimax.id_minimax(to_board(p, o, color), color, depth)
    return r * 8 + c


def play_game(agent, agent_color, opponent, record=True, greedy=False, random_plies=0):
    """
    opponent: ("random"|"fast"|"minimax", depth, p_random) hoặc ("self", model).
    random_plies: số nước đầu ván đi ngẫu nhiên (cả hai bên), không ghi vào dữ liệu train.
    Trả về kết quả theo góc nhìn agent (>0 thắng, <0 thua, 0 hòa).
    Nếu record=True, lưu dữ liệu train của các bên do mạng điều khiển.
    """
    discs = dict(START)
    turn = BLACK
    ply = 0
    traj = {BLACK: ([], []), WHITE: ([], [])}
    while True:
        p, o = discs[turn], discs[3 - turn]
        if not legal_moves(p, o):
            if not legal_moves(o, p):
                break
            turn = 3 - turn  # mất lượt
            continue

        if ply < random_plies:
            sq = random.choice(bits(legal_moves(p, o)))
        elif turn == agent_color:
            sq, v = agent.choose(p, o, greedy=greedy)
            traj[turn][0].append(encode(p, o)); traj[turn][1].append(v)
        elif opponent[0] == "self":
            sq, v = agent.choose(p, o, greedy=greedy, model=opponent[1])
            traj[turn][0].append(encode(p, o)); traj[turn][1].append(v)
        else:
            sq = opponent_move(p, o, turn, *opponent)

        discs[turn], discs[3 - turn] = apply_move(p, o, sq)
        turn = 3 - turn
        ply += 1

    if record:
        for color, (states, values) in traj.items():
            if states:
                agent.store_game(states, values, final_value(discs[color], discs[3 - color]))
    return final_value(discs[agent_color], discs[3 - agent_color])


def evaluate(agent, opponent, n_games):
    wins = draws = 0
    for i in range(n_games):
        z = play_game(agent, BLACK if i % 2 == 0 else WHITE, opponent, record=False, greedy=True,
                      random_plies=OPENING_RANDOM_PLIES)
        wins += z > 0
        draws += z == 0
    return wins / n_games, draws / n_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval", type=int, default=0, help="chỉ đánh giá model đã lưu, N ván mỗi đối thủ")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=int, default=0, help="level curriculum bắt đầu (0..%d)" % (len(LEVELS) - 1))
    parser.add_argument("--epsilon", type=float, default=None, help="epsilon ban đầu (mặc định 0.3)")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs("models", exist_ok=True)

    agent = MinimaxQAgent()
    if args.epsilon is not None:
        agent.epsilon = args.epsilon
    if (args.resume or args.eval) and os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print("✅ Model loaded from", MODEL_PATH)

    eval_opponents = [("random", 0, 1.0), ("fast", 2, 0.0), ("fast", 4, 0.0),
                      ("minimax", 1, 0.0), ("minimax", 2, 0.0), ("minimax", 3, 0.0)]
    if args.eval:
        for opp in eval_opponents:
            w, d = evaluate(agent, opp, args.eval)
            print(f"vs {opp[0]} depth={opp[1]}: thắng {w:.0%}, hòa {d:.0%}")
        return

    metrics = TrainingMetrics()
    level = min(args.level, len(LEVELS) - 1)
    recent = deque(maxlen=PROMOTE_WINDOW)
    all_results = deque(maxlen=100)
    snapshot = copy.deepcopy(agent.model).eval()
    start = time.time()

    for game in range(1, args.games + 1):
        agent_color = BLACK if game % 2 else WHITE
        self_play = random.random() < SELF_PLAY_RATIO
        opponent = ("self", snapshot) if self_play else LEVELS[level]

        z = play_game(agent, agent_color, opponent)
        for _ in range(TRAIN_STEPS_PER_GAME):
            agent.replay(256)
        agent.decay_epsilon()

        if not self_play:
            recent.append(z > 0)
            all_results.append(z > 0)
            if (len(recent) == PROMOTE_WINDOW and np.mean(recent) >= PROMOTE_WIN_RATE
                    and level < len(LEVELS) - 1):
                level += 1
                recent.clear()
                print(f"⬆️  Game {game}: lên level {level} -> {LEVELS[level]}")

        if game % SNAPSHOT_EVERY == 0:
            snapshot = copy.deepcopy(agent.model).eval()

        metrics.update(
            win_rate=float(np.mean(all_results)) if all_results else 0.0,
            loss=agent.current_loss,
            epsilon=agent.epsilon,
            reward=z,
        )

        if game % 50 == 0:
            print(f"game {game} | level {level} {LEVELS[level]} | win(50) {np.mean(recent) if recent else 0:.2f}"
                  f" | loss {agent.current_loss:.4f} | eps {agent.epsilon:.3f} | {time.time() - start:.0f}s",
                  flush=True)
        if game % args.eval_every == 0:
            agent.save(MODEL_PATH)
            metrics.plot()
            for opp in eval_opponents[1:3]:
                w, d = evaluate(agent, opp, 20)
                print(f"   eval vs {opp[0]} depth={opp[1]}: thắng {w:.0%}, hòa {d:.0%}", flush=True)

    agent.save(MODEL_PATH)
    metrics.plot()
    print("Đã lưu model tại", MODEL_PATH)


if __name__ == "__main__":
    main()
