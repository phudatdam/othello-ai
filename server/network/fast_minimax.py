"""
Minimax alpha-beta chạy trên bitboard, dùng làm đối thủ khi train.
Minimax gốc (ai/minimax.py) chạy trên list và evaluator đầy đủ nên rất chậm
(khoảng 13 giây cho một ván ở độ sâu 2), bản này nhanh hơn nhiều lần.
Hàm đánh giá: bảng trọng số vị trí + độ linh động (số nước đi).
"""
import random

from network.minimax_q_learning import apply_move, bits, legal_moves

INF = 1_000_000

_W = [
    100, -20, 10, 5, 5, 10, -20, 100,
    -20, -50, -2, -2, -2, -2, -50, -20,
    10, -2, -1, -1, -1, -1, -2, 10,
    5, -2, -1, -1, -1, -1, -2, 5,
    5, -2, -1, -1, -1, -1, -2, 5,
    10, -2, -1, -1, -1, -1, -2, 10,
    -20, -50, -2, -2, -2, -2, -50, -20,
    100, -20, 10, 5, 5, 10, -20, 100,
]


def _positional(b):
    return sum(_W[sq] for sq in bits(b))


def evaluate(p, o):
    mp = bin(legal_moves(p, o)).count("1")
    mo = bin(legal_moves(o, p)).count("1")
    return _positional(p) - _positional(o) + 5 * (mp - mo)


def negamax(p, o, depth, alpha, beta):
    moves = legal_moves(p, o)
    if not moves:
        if not legal_moves(o, p):
            diff = bin(p).count("1") - bin(o).count("1")
            return INF * (diff > 0) - INF * (diff < 0)
        return -negamax(o, p, depth, -beta, -alpha)  # mất lượt, không trừ độ sâu
    if depth == 0:
        return evaluate(p, o)
    best = -INF - 1
    for sq in bits(moves):
        np_, no = apply_move(p, o, sq)
        v = -negamax(no, np_, depth - 1, -beta, -alpha)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def best_move(p, o, depth):
    moves = bits(legal_moves(p, o))
    random.shuffle(moves)  # phá thế hòa ngẫu nhiên để các ván không giống hệt nhau
    best, best_v = moves[0], -INF - 1
    alpha = -INF - 1
    for sq in moves:
        np_, no = apply_move(p, o, sq)
        v = -negamax(no, np_, depth - 1, -INF - 1, -alpha)
        if v > best_v:
            best, best_v = sq, v
        alpha = max(alpha, v)
    return best
