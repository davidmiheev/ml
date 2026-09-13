"""Infinite-bandwidth upper bound on the objective.

Drop the server and house budgets and only keep the per-device rollout budget.
Then for class c the best happiness available on day t is the non-decreasing
B_c(t) = max{h_k : c_k = c, r_k <= t}. A device's class-c slot holds a step
function that starts at 0, changes at most q times and whose value on a step
beginning at day t is at most B_c(t); since B_c is non-decreasing, the best such
function steps exactly at the "records" (running maxima of h by release day) at
their own release days. That best chain value F_c(q) does not depend on the
device -- device i just scales it by w[i][c] -- so it is computed once per class
and then split across classes per device under the budget u_i.

Usage:  python3 tools/bound.py <input_file>
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import score as scorer  # noqa: E402


def records(inp, c):
    """(release day, happiness) of the running maxima of class c, by day."""
    best_by_day = {}
    for k in range(inp["U"]):
        if inp["cls"][k] != c:
            continue
        r, h = inp["rel"][k], inp["happy"][k]
        if h > best_by_day.get(r, 0):
            best_by_day[r] = h
    out, running = [], 0
    for r in sorted(best_by_day):
        if best_by_day[r] > running:
            running = best_by_day[r]
            out.append((r, running))
    return out


def chain_values(inp, c, qmax):
    """F_c(q) for q = 0..qmax: best chain value using at most q rollouts."""
    recs = records(inp, c)
    T = inp["T"]
    n = len(recs)
    F = [0] * (qmax + 1)
    if n == 0 or qmax == 0:
        return F
    # dp[q][j]: best value when exactly q records are used and j is the last one.
    NEG = float("-inf")
    dp = [[NEG] * n for _ in range(qmax + 1)]
    for j in range(n):
        r, h = recs[j]
        dp[1][j] = h * (T - r)
    for q in range(2, qmax + 1):
        for j in range(n):
            r, h = recs[j]
            best = NEG
            for p in range(j):
                if dp[q - 1][p] == NEG:
                    continue
                cand = dp[q - 1][p] + (h - recs[p][1]) * (T - r)
                if cand > best:
                    best = cand
            dp[q][j] = best
    for q in range(1, qmax + 1):
        top = max([v for v in dp[q] if v != NEG], default=NEG)
        F[q] = max(F[q - 1], top if top != NEG else F[q - 1])
    return F


def bound(inp):
    C, S = inp["C"], inp["S"]
    qmax = max(inp["cap"]) if S else 0
    F = [chain_values(inp, c, qmax) for c in range(C)]
    total = 0
    for i in range(S):
        u = inp["cap"][i]
        w = inp["w"][i]
        # Split u slots across classes; marginal gains per class are concave in
        # practice, but with small u an exact DP is cheap and safe.
        best = [0] * (u + 1)
        for c in range(C):
            nxt = [0] * (u + 1)
            for used in range(u + 1):
                v = 0
                for q in range(used + 1):
                    cand = best[used - q] + w[c] * F[c][q]
                    if cand > v:
                        v = cand
                nxt[used] = v
            best = nxt
        total += best[u]
    return total


if __name__ == "__main__":
    print(bound(scorer.read_input(sys.argv[1])))
