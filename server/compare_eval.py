"""
Thí nghiệm đối chứng: hàm đánh giá học bằng Q-learning (mạng) vs evaluator viết tay.

Hai bên dùng CÙNG thuật toán: minimax đầy đủ (full-width) trên bitboard, cùng độ sâu,
cùng cách xử lý mất lượt và kết thúc ván. Chỉ khác hàm đánh giá ở lá:
  - "net":  mạng MinimaxQNetwork (models/minimax_q_cnn.pt)
  - "hand": ai/evaluator.py của project (hàm mà minimax gốc dùng)

Mỗi khai cuộc ngẫu nhiên (vài nước đầu) được chơi 2 lần, đổi màu, để loại may rủi khai cuộc.

Chạy:  python compare_eval.py --depths 1 2 3 --pairs 100 --workers 4
"""
import argparse
import math
import random
import time
from multiprocessing import Pool

import numpy as np
import torch

from ai import evaluator
from network.minimax_q_learning import (
    BLACK, WHITE, MinimaxQAgent, apply_move, bits, encode, legal_moves, to_board,
)

MODEL_PATH = "models/minimax_q_cnn.pt"
START = {BLACK: (1 << 28) | (1 << 35), WHITE: (1 << 27) | (1 << 36)}
WIN = 1e9

_agent = None


def _init_worker(seed):
    global _agent
    torch.set_num_threads(1)
    random.seed(seed)
    _agent = MinimaxQAgent()
    _agent.load(MODEL_PATH)
    _agent.model.eval()


def _collect_leaves(p, o, root_is_p, depth, leaves):
    """
    Dựng cây minimax đầy đủ. p là người sắp đi.
    root_is_p: người sắp đi có phải người ở gốc không.
    Trả về cây dạng ("max"/"min", [con]) | ("leaf", idx) | ("term", value).
    """
    moves = legal_moves(p, o)
    if not moves:
        if not legal_moves(o, p):
            diff = bin(p).count("1") - bin(o).count("1")
            v = WIN * np.sign(diff)
            return ("term", v if root_is_p else -v)
        return _collect_leaves(o, p, not root_is_p, depth, leaves)  # mất lượt, không tính độ sâu
    if depth == 0:
        leaves.append((p, o, root_is_p))
        return ("leaf", len(leaves) - 1)
    kids = []
    for sq in bits(moves):
        np_, no = apply_move(p, o, sq)
        kids.append(_collect_leaves(no, np_, not root_is_p, depth - 1, leaves))
    return ("max" if root_is_p else "min", kids)


def _backup(node, vals):
    kind, x = node
    if kind == "leaf":
        return vals[x]
    if kind == "term":
        return x
    vs = [_backup(k, vals) for k in x]
    return max(vs) if kind == "max" else min(vs)


def _leaf_values(kind, leaves, root_color):
    """Giá trị lá theo góc nhìn người ở gốc."""
    if kind == "net":
        x = torch.from_numpy(np.stack([encode(p, o) for p, o, _ in leaves]))
        with torch.no_grad():
            v = _agent.model(x).numpy()  # góc nhìn người sắp đi
        return [float(val) if r else -float(val) for val, (_, _, r) in zip(v, leaves)]
    out = []
    for p, o, r in leaves:
        mover = root_color if r else 3 - root_color
        board = to_board(p, o, mover)
        out.append(evaluator.evaluate(board, root_color))
    return out


def search_move(kind, p, o, color, depth):
    best, best_v = None, -math.inf
    moves = bits(legal_moves(p, o))
    random.shuffle(moves)
    trees, leaves = [], []
    for sq in moves:
        np_, no = apply_move(p, o, sq)
        trees.append((sq, _collect_leaves(no, np_, False, depth - 1, leaves)))
    vals = _leaf_values(kind, leaves, color) if leaves else []
    for sq, t in trees:
        v = _backup(t, vals)
        if v > best_v:
            best, best_v = sq, v
    return best


def play(job):
    """job = (opening_seed, depth, net_color, opening_plies) -> (kết quả cho net, thời gian net, thời gian hand)"""
    seed, depth, net_color, opening_plies = job
    rng = random.Random(seed)
    discs = dict(START)
    turn, ply = BLACK, 0
    t_used = {"net": [0.0, 0], "hand": [0.0, 0]}
    while True:
        p, o = discs[turn], discs[3 - turn]
        if not legal_moves(p, o):
            if not legal_moves(o, p):
                break
            turn = 3 - turn
            continue
        if ply < opening_plies:
            sq = rng.choice(bits(legal_moves(p, o)))
        else:
            kind = "net" if turn == net_color else "hand"
            t = time.time()
            sq = search_move(kind, p, o, turn, depth)
            t_used[kind][0] += time.time() - t
            t_used[kind][1] += 1
        discs[turn], discs[3 - turn] = apply_move(p, o, sq)
        turn = 3 - turn
        ply += 1
    diff = bin(discs[net_color]).count("1") - bin(discs[3 - net_color]).count("1")
    score = 1.0 if diff > 0 else 0.0 if diff < 0 else 0.5
    return score, t_used


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--pairs", type=int, default=100, help="số khai cuộc; mỗi khai cuộc chơi 2 ván đổi màu")
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    with Pool(args.workers, initializer=_init_worker, initargs=(args.seed,)) as pool:
        for depth in args.depths:
            jobs = [(args.seed + i, depth, c, args.opening_plies)
                    for i in range(args.pairs) for c in (BLACK, WHITE)]
            start = time.time()
            res = pool.map(play, jobs, chunksize=1)
            scores = [s for s, _ in res]
            wins = sum(s == 1.0 for s in scores)
            draws = sum(s == 0.5 for s in scores)
            losses = sum(s == 0.0 for s in scores)
            n = len(scores)
            pts = sum(scores)
            lo, hi = wilson(pts, n)
            tn = sum(t["net"][0] for _, t in res) / max(1, sum(t["net"][1] for _, t in res))
            th = sum(t["hand"][0] for _, t in res) / max(1, sum(t["hand"][1] for _, t in res))
            print(f"độ sâu {depth}: {n} ván | net thắng {wins}, hòa {draws}, thua {losses} | "
                  f"điểm net {pts / n:.1%} (KTC 95%: {lo:.1%} - {hi:.1%}) | "
                  f"thời gian/nước: net {tn * 1000:.1f} ms, hand {th * 1000:.1f} ms | {time.time() - start:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
