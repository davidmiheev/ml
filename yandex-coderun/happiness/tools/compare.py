"""Compare a scenario's solution against the infinite-bandwidth upper bound.

Usage:  python3 tools/compare.py <scenario> [seeds...]
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gen, score as scorer, bound as bnd  # noqa: E402

sc = int(sys.argv[1]); seeds = [int(x) for x in sys.argv[2:]] or list(range(8))
ratios = []
for seed in seeds:
    path = os.path.join(ROOT, "tests", "cmp.in")
    open(path, "w").write(gen.gen(sc, seed))
    inp = scorer.read_input(path)
    st = time.time()
    with open(path) as fin:
        res = subprocess.run([sys.executable, os.path.join(ROOT, "solutions", "scenario_%d.py" % sc)],
                             stdin=fin, capture_output=True, text=True)
    el = time.time() - st
    tok = res.stdout.split()
    plan = [(int(tok[1+3*j]), int(tok[2+3*j]), int(tok[3+3*j])) for j in range(int(tok[0]))]
    got = scorer.score(inp, plan)
    ub = bnd.bound(inp)
    ratios.append(got / ub)
    print("seed %-3d score %12d   bound %12d   %6.2f%% of bound   %5.2fs  (%d rollouts)"
          % (seed, got, ub, 100.0 * got / ub, el, len(plan)))
print("-- mean %.2f%% of the upper bound" % (100.0 * sum(ratios) / len(ratios)))
