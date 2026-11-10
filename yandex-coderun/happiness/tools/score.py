"""Validate a rollout plan against the problem constraints and score it.

Usage:  python3 tools/score.py <input_file> <output_file>
Exits non-zero and prints the violated rule if the plan is invalid.
"""
import sys
from collections import defaultdict


def read_input(path):
    with open(path) as f:
        tok = f.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(tok[p])
        p += 1
        return v

    T, S, U, C, H, G = (nxt() for _ in range(6))
    limit = [nxt() for _ in range(H)]
    house, cap, w = [], [], []
    for _ in range(S):
        house.append(nxt())
        cap.append(nxt())
        w.append([nxt() for _ in range(C)])
    cls, size, happy, rel = [], [], [], []
    for _ in range(U):
        cls.append(nxt())
        size.append(nxt())
        happy.append(nxt())
        rel.append(nxt())
    return dict(T=T, S=S, U=U, C=C, H=H, G=G, limit=limit, house=house,
                cap=cap, w=w, cls=cls, size=size, happy=happy, rel=rel)


def read_output(path):
    with open(path) as f:
        tok = f.read().split()
    n = int(tok[0])
    if len(tok) != 1 + 3 * n:
        raise ValueError("declared %d rollouts but found %d numbers after the count"
                         % (n, len(tok) - 1))
    return [(int(tok[1 + 3 * j]), int(tok[2 + 3 * j]), int(tok[3 + 3 * j]))
            for j in range(n)]


def score(inp, plan):
    """Return total weighted happiness, or raise ValueError on any violation."""
    T, S, U, C = inp["T"], inp["S"], inp["U"], inp["C"]
    cls, size, happy, rel = inp["cls"], inp["size"], inp["happy"], inp["rel"]
    house, cap, w, limit, G = (inp["house"], inp["cap"], inp["w"],
                               inp["limit"], inp["G"])

    seen_dev_upd = set()       # (device, update) may appear at most once overall
    seen_day_dev_cls = set()   # one update per (day, device, class)
    per_device = defaultdict(int)
    per_day = defaultdict(int)
    per_day_house = defaultdict(int)
    chains = defaultdict(list)  # (device, class) -> [(day, happiness)]

    for t, i, k in plan:
        if not (0 <= t < T):
            raise ValueError("day %d out of range" % t)
        if not (0 <= i < S):
            raise ValueError("device %d out of range" % i)
        if not (0 <= k < U):
            raise ValueError("update %d out of range" % k)
        if t < rel[k]:
            raise ValueError("update %d rolled out on day %d before release day %d"
                             % (k, t, rel[k]))
        if (i, k) in seen_dev_upd:
            raise ValueError("update %d rolled out to device %d more than once" % (k, i))
        seen_dev_upd.add((i, k))
        key = (t, i, cls[k])
        if key in seen_day_dev_cls:
            raise ValueError("two updates of class %d on device %d on day %d"
                             % (cls[k], i, t))
        seen_day_dev_cls.add(key)

        per_device[i] += 1
        if per_device[i] > cap[i]:
            raise ValueError("device %d exceeds its budget of %d rollouts" % (i, cap[i]))
        per_day[t] += size[k]
        if per_day[t] > G:
            raise ValueError("day %d exceeds the server budget: %d > %d"
                             % (t, per_day[t], G))
        m = house[i]
        per_day_house[(t, m)] += size[k]
        if per_day_house[(t, m)] > limit[m]:
            raise ValueError("day %d house %d exceeds its budget: %d > %d"
                             % (t, m, per_day_house[(t, m)], limit[m]))
        chains[(i, cls[k])].append((t, happy[k]))

    total = 0
    for (i, j), events in chains.items():
        events.sort()
        weight = w[i][j]
        for a, (day, h) in enumerate(events):
            end = events[a + 1][0] if a + 1 < len(events) else T
            total += weight * h * (end - day)
    return total


def main():
    inp = read_input(sys.argv[1])
    plan = read_output(sys.argv[2])
    try:
        print(score(inp, plan))
    except ValueError as e:
        print("INVALID: %s" % e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
