"""Run a scenario's solution over generated tests and report the scores.

Usage:  python3 tools/run.py <scenario 1..6> [seeds...]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import gen  # noqa: E402
import score as scorer  # noqa: E402


def run(scenario, seeds):
    solution = os.path.join(ROOT, "solutions", "scenario_%d.py" % scenario)
    results = []
    for seed in seeds:
        test = gen.gen(scenario, seed)
        in_path = os.path.join(ROOT, "tests", "s%d_%d.in" % (scenario, seed))
        out_path = in_path[:-3] + ".out"
        with open(in_path, "w") as f:
            f.write(test)

        started = time.time()
        with open(in_path) as fin, open(out_path, "w") as fout:
            proc = subprocess.run([sys.executable, solution], stdin=fin, stdout=fout)
        elapsed = time.time() - started
        if proc.returncode != 0:
            print("seed %-3d CRASHED (exit %d)" % (seed, proc.returncode))
            results.append(None)
            continue

        inp = scorer.read_input(in_path)
        try:
            value = scorer.score(inp, scorer.read_output(out_path))
        except ValueError as e:
            print("seed %-3d INVALID: %s" % (seed, e))
            results.append(None)
            continue
        print("seed %-3d score %14d   %6.2fs" % (seed, value, elapsed))
        results.append(value)
    good = [r for r in results if r is not None]
    if len(good) == len(results) and good:
        print("-- all %d valid, mean %.1f" % (len(good), sum(good) / len(good)))
    return results


if __name__ == "__main__":
    sc = int(sys.argv[1])
    sd = [int(x) for x in sys.argv[2:]] or list(range(8))
    run(sc, sd)
