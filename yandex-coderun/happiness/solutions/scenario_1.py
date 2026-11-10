"""Scenario 1 - "Proverka svyazi": T=1, S=1, U=1, C=1, H=1.

There is exactly one candidate rollout in the entire problem: update 0 onto
device 0 on day 0. Its value is w[0][0] * h[0] * (T - 0) > 0, so shipping it is
optimal whenever it is legal. We still check legality against the input rather
than assuming it, so a test that deviates from the advertised constraints gets
an empty (but valid) plan instead of a rejected one.
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
        day, dev, upd = 0, 0, 0
        fits = (release[upd] <= day
                and budget[dev] >= 1
                and size[upd] <= G
                and size[upd] <= limit[house[dev]])
        if fits and weight[dev][cls[upd]] * happy[upd] > 0:
            plan.append((day, dev, upd))

    out = [str(len(plan))]
    out.extend("%d %d %d" % row for row in plan)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
