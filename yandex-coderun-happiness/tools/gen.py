"""Generate random tests matching each scenario's stated constraints.

Usage:  python3 tools/gen.py <scenario 1..6> <seed> [> test.in]

The statement fixes T, S, U, C, H, G exactly and gives ranges for the rest.
Release days and house occupancy are unspecified, so each scenario draws them
from a few different shapes (picked by the seed) to keep solutions honest:
releases may be uniform or clustered on "Friday", houses may be evenly
populated or heavily skewed.
"""
import random
import sys

# T, S, U, C, H, G, l range, u range, w range, s range, h range
SPECS = {
    1: (1, 1, 1, 1, 1, 1000, (1000, 1000), (1, 1), (1, 30), (5, 40), (1, 100)),
    2: (3, 4, 1, 1, 1, 40, (1000, 1000), (1, 1), (1, 30), (40, 40), (1, 100)),
    3: (14, 8, 30, 2, 2, 100, (50, 50), (2, 4), (1, 30), (20, 50), (1, 100)),
    4: (30, 50, 100, 5, 10, 500, (50, 100), (3, 8), (1, 50), (10, 100), (1, 100)),
    5: (90, 1000, 500, 10, 100, 5000, (20, 200), (5, 15), (1, 100), (10, 200), (1, 100)),
    6: (365, 100000, 10000, 30, 10000, 1000000, (10, 1000), (5, 30), (0, 100), (1, 1000), (1, 100)),
}


def gen(scenario, seed):
    T, S, U, C, H, G, lr, ur, wr, sr, hr = SPECS[scenario]
    rng = random.Random(seed)
    # Shape choices, so a solution is not tuned to one particular distribution.
    clustered_releases = bool(seed % 2)
    skewed_houses = bool((seed // 2) % 2)

    out = ["%d %d %d %d %d %d" % (T, S, U, C, H, G)]
    out.append(" ".join(str(rng.randint(*lr)) for _ in range(H)))

    if skewed_houses and H > 1:
        # A few houses hold many devices, most hold one or two.
        weights = [rng.random() ** 3 for _ in range(H)]
        total = sum(weights)
        houses = rng.choices(range(H), weights=[x / total for x in weights], k=S)
    else:
        houses = [rng.randrange(H) for _ in range(S)]

    for i in range(S):
        row = [houses[i], rng.randint(*ur)]
        row.extend(rng.randint(*wr) for _ in range(C))
        out.append(" ".join(map(str, row)))

    for _ in range(U):
        if clustered_releases and T >= 7:
            # Most releases land on a "Friday", the rest are spread out.
            if rng.random() < 0.6:
                fridays = list(range(4, T, 7))
                r = rng.choice(fridays) if fridays else rng.randrange(T)
            else:
                r = rng.randrange(T)
        else:
            r = rng.randrange(T)
        out.append("%d %d %d %d" % (rng.randrange(C), rng.randint(*sr),
                                    rng.randint(*hr), r))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.write(gen(int(sys.argv[1]), int(sys.argv[2])))
