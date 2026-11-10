"""Scenario 2 - "CDN na kolenke": T=3, S=4, U=1, C=1, H=1, G=40, every s_k=40.

The server budget equals one model size, so exactly one rollout fits per day;
the house limit (1000) is slack. With a single update and u_i=1, every device
can receive it at most once, so the plan is just an assignment of devices to
distinct days. A device served on day t is worth w[i][c0] * h0 * (T - t), i.e.
weight times a day multiplier, so by the rearrangement inequality the optimum
pairs the largest weights with the largest multipliers: sort devices by weight
descending and hand out the earliest legal days in that order.

The day search below also honours the per-house limit, so the same code stays
correct (and stays optimal for a single house) if the budgets are less tight
than advertised.
"""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        value = int(data[pos])
        pos += 1
        return value

    T, S, U, C, H, G = (nxt() for _ in range(6))
    limit = [nxt() for _ in range(H)]

    house, budget, weight = [], [], []
    for _ in range(S):
        house.append(nxt())
        budget.append(nxt())
        weight.append([nxt() for _ in range(C)])

    cls, size, happy, release = [], [], [], []
    for _ in range(U):
        cls.append(nxt())
        size.append(nxt())
        happy.append(nxt())
        release.append(nxt())

    plan = []
    if T > 0 and S > 0 and U > 0:
        upd = 0
        c, s, h, r = cls[upd], size[upd], happy[upd], release[upd]

        # Devices worth serving at all, best first.
        order = [i for i in range(S)
                 if budget[i] >= 1 and weight[i][c] * h > 0 and s <= limit[house[i]]]
        order.sort(key=lambda i: weight[i][c], reverse=True)

        free_server = [G] * T
        free_house = [[limit[m] for m in range(H)] for _ in range(T)]
        for i in order:
            m = house[i]
            for t in range(r, T):
                if free_server[t] >= s and free_house[t][m] >= s:
                    free_server[t] -= s
                    free_house[t][m] -= s
                    plan.append((t, i, upd))
                    break

    out = [str(len(plan))]
    out.extend("%d %d %d" % row for row in plan)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
