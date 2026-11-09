"""Exhaustive optimum for scenario 2, used to certify solutions/scenario_2.py.

Every device independently either skips the update or takes one of the T days,
which is (T+1)**S plans; each is validated and scored by tools/score.py so the
check cannot drift away from the real rules.
"""
import itertools
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gen, score as scorer  # noqa: E402


def brute(inp):
    T, S = inp["T"], inp["S"]
    best, best_plan = 0, []
    for choice in itertools.product(range(-1, T), repeat=S):
        plan = [(t, i, 0) for i, t in enumerate(choice) if t >= 0]
        try:
            value = scorer.score(inp, plan)
        except ValueError:
            continue
        if value > best:
            best, best_plan = value, plan
    return best, best_plan


for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 12):
    path = os.path.join(ROOT, "tests", "brute2.in")
    with open(path, "w") as f:
        f.write(gen.gen(2, seed))
    inp = scorer.read_input(path)
    with open(path) as fin:
        got = subprocess.run([sys.executable, os.path.join(ROOT, "solutions", "scenario_2.py")],
                             stdin=fin, capture_output=True, text=True).stdout
    tok = got.split()
    plan = [(int(tok[1 + 3 * j]), int(tok[2 + 3 * j]), int(tok[3 + 3 * j]))
            for j in range(int(tok[0]))]
    mine = scorer.score(inp, plan)
    best, _ = brute(inp)
    print("seed %-3d mine %8d  optimum %8d  %s"
          % (seed, mine, best, "OK" if mine == best else "*** SUBOPTIMAL ***"))
