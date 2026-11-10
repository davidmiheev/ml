"""Scenario 3 - "Dve nedeli bez dostupov": T=14, S=8, U=30, C=2, H=2.

Small instance (at most ~32 rollouts fit), so it is worth searching properly
rather than settling for a greedy plan.

The objective telescopes: if a (device, class) slot receives updates with
happiness h_1..h_p on days t_1 < ... < t_p, it contributes
w * sum_a h_a * (t_{a+1} - t_a) with t_{p+1} = T. So a single rollout's
marginal value is w[i][c_k] * (h_k - h_on_slot) * (T - t) -- every rollout is
worth more the earlier it ships, and only beats the status quo when h_k tops
what the slot already holds.

Both budgets here are tight and comparable: the house limit (50/day, sizes
20..50) allows one or two rollouts per house per day, while the device budgets
u_i = 2..4 allow 16..32 in total. Because u_i is the only *global* resource,
day-by-day greed misjudges it -- spending a slot on an early mediocre model can
lock out a much better one released later. So a greedy plan only seeds the
search, and simulated annealing over add/remove/move/replace moves does the
rest, with every move evaluated incrementally against the affected slot alone.
"""
import random
import sys
import time

TIME_BUDGET = 2.0  # seconds of search; the greedy seed alone is already valid


def main():
    started = time.time()
    data = sys.stdin.buffer.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        value = int(data[pos])
        pos += 1
        return value

    T, S, U, C, H, G = (nxt() for _ in range(6))
    limit = [nxt() for _ in range(H)]

    house, cap, weight = [0] * S, [0] * S, []
    for i in range(S):
        house[i] = nxt()
        cap[i] = nxt()
        weight.append([nxt() for _ in range(C)])

    cls, size, happy, release = [0] * U, [0] * U, [0] * U, [0] * U
    for k in range(U):
        cls[k] = nxt()
        size[k] = nxt()
        happy[k] = nxt()
        release[k] = nxt()

    by_class = [[] for _ in range(C)]
    for k in range(U):
        by_class[cls[k]].append(k)
    # Best first: a slot's chain only ever climbs in happiness.
    for c in range(C):
        by_class[c].sort(key=lambda k: -happy[k])

    # ---- state -------------------------------------------------------------
    chain = [[] for _ in range(S * C)]   # (day, update) per slot, sorted by day
    val = [0] * (S * C)                  # weighted value of that slot
    used_dev = [0] * S
    on_dev = [set() for _ in range(S)]
    used_srv = [0] * T
    used_house = [[0] * H for _ in range(T)]
    total = 0

    def slot_value(i, c, lst):
        if not lst:
            return 0
        acc = 0
        last = T
        for a in range(len(lst) - 1, -1, -1):
            t, k = lst[a]
            acc += happy[k] * (last - t)
            last = t
        return acc * weight[i][c]

    def can_add(t, i, k):
        if used_dev[i] >= cap[i] or t < release[k] or k in on_dev[i]:
            return False
        if used_srv[t] + size[k] > G:
            return False
        m = house[i]
        if used_house[t][m] + size[k] > limit[m]:
            return False
        for dt, _ in chain[i * C + cls[k]]:
            if dt == t:
                return False
        return True

    def apply_add(t, i, k):
        nonlocal total
        sl = i * C + cls[k]
        lst = chain[sl]
        lst.append((t, k))
        lst.sort()
        new = slot_value(i, cls[k], lst)
        total += new - val[sl]
        val[sl] = new
        used_dev[i] += 1
        on_dev[i].add(k)
        used_srv[t] += size[k]
        used_house[t][house[i]] += size[k]

    def apply_remove(t, i, k):
        nonlocal total
        sl = i * C + cls[k]
        lst = chain[sl]
        lst.remove((t, k))
        new = slot_value(i, cls[k], lst)
        total += new - val[sl]
        val[sl] = new
        used_dev[i] -= 1
        on_dev[i].discard(k)
        used_srv[t] -= size[k]
        used_house[t][house[i]] -= size[k]

    def gain_of_add(t, i, k):
        """Exact change in the objective from adding (t, i, k)."""
        sl = i * C + cls[k]
        lst = chain[sl]
        return slot_value(i, cls[k], sorted(lst + [(t, k)])) - val[sl]

    # ---- relaxation-guided construction -----------------------------------
    # Drop both bandwidth limits: the best happiness of class c available on day
    # t, B_c(t) = max{h_k : c_k = c, r_k <= t}, is non-decreasing, so the best
    # chain of q rollouts steps at the "records" (running maxima of h by release
    # day) on their own release days. That chain value F_c(q) does not depend on
    # the device -- device i only scales it by w[i][c] -- so it is computed once
    # per class and then split across classes under the budget u_i. Scheduling
    # those picks at the earliest feasible day gives a plan that is already right
    # whenever bandwidth is slack and the device budget is what binds.
    NEG = float("-inf")
    qcap = max(cap) if S else 0
    records = []
    for c in range(C):
        best_by_day = {}
        for k in by_class[c]:
            if happy[k] > best_by_day.get(release[k], (0, -1))[0]:
                best_by_day[release[k]] = (happy[k], k)
        recs, running = [], 0
        for r in sorted(best_by_day):
            h, k = best_by_day[r]
            if h > running:
                running = h
                recs.append((r, h, k))
        records.append(recs)

    # F[c][q] with the chain that attains it.
    F, Fpick = [], []
    for c in range(C):
        recs = records[c]
        n = len(recs)
        f = [0] * (qcap + 1)
        pick = [[] for _ in range(qcap + 1)]
        if n and qcap:
            dp = [[NEG] * n for _ in range(qcap + 1)]
            par = [[-1] * n for _ in range(qcap + 1)]
            for j in range(n):
                dp[1][j] = recs[j][1] * (T - recs[j][0])
            for q in range(2, qcap + 1):
                for j in range(n):
                    r, h, _ = recs[j]
                    for p in range(j):
                        if dp[q - 1][p] == NEG:
                            continue
                        cand = dp[q - 1][p] + (h - recs[p][1]) * (T - r)
                        if cand > dp[q][j]:
                            dp[q][j] = cand
                            par[q][j] = p
            for q in range(1, qcap + 1):
                bj, bv = -1, NEG
                for j in range(n):
                    if dp[q][j] > bv:
                        bv, bj = dp[q][j], j
                if bj >= 0 and bv > f[q - 1]:
                    f[q] = bv
                    seq, j, qq = [], bj, q
                    while j >= 0 and qq >= 1:
                        seq.append(recs[j][2])
                        j = par[qq][j]
                        qq -= 1
                    pick[q] = seq[::-1]
                else:
                    f[q] = f[q - 1]
                    pick[q] = list(pick[q - 1])
        F.append(f)
        Fpick.append(pick)

    # Split each device's budget across classes, then schedule best slots first.
    wanted = []
    for i in range(S):
        u = cap[i]
        row = [[0] * (u + 1) for _ in range(C + 1)]
        choice = [[0] * (u + 1) for _ in range(C + 1)]
        for c in range(C):
            for used in range(u + 1):
                bv, bq = -1, 0
                for q in range(used + 1):
                    cand = row[c][used - q] + weight[i][c] * F[c][q]
                    if cand > bv:
                        bv, bq = cand, q
                row[c + 1][used], choice[c + 1][used] = bv, bq
        left = u
        for c in range(C - 1, -1, -1):
            q = choice[c + 1][left]
            if q:
                wanted.append((weight[i][c] * F[c][q], i, c, Fpick[c][q]))
            left -= q
    wanted.sort(reverse=True, key=lambda x: x[0])

    for _, i, c, seq in wanted:
        day = 0
        for k in seq:
            t = max(day, release[k])
            while t < T and not can_add(t, i, k):
                t += 1
            if t >= T:
                break
            apply_add(t, i, k)
            day = t + 1
    relaxed_chain, relaxed_total = [list(x) for x in chain], total
    for sl in range(S * C):
        chain[sl] = []
        val[sl] = 0
    for i in range(S):
        used_dev[i] = 0
        on_dev[i].clear()
    for t in range(T):
        used_srv[t] = 0
        for m in range(H):
            used_house[t][m] = 0
    total = 0

    # ---- greedy seed: best gain per megabyte, earliest days first ----------
    for t in range(T):
        while True:
            best, best_ratio = None, 0.0
            for i in range(S):
                if used_dev[i] >= cap[i]:
                    continue
                for c in range(C):
                    if weight[i][c] == 0:
                        continue
                    for k in by_class[c]:
                        if not can_add(t, i, k):
                            continue
                        g = gain_of_add(t, i, k)
                        if g <= 0:
                            continue
                        ratio = g / size[k]
                        if ratio > best_ratio:
                            best_ratio, best = ratio, (t, i, k)
            if best is None:
                break
            apply_add(*best)

    def snapshot():
        return [list(x) for x in chain]

    greedy_chain, greedy_total = [list(x) for x in chain], total

    def restore(snap):
        """Rebuild the whole state from a snapshot of the slot chains."""
        nonlocal total
        for sl in range(S * C):
            chain[sl] = []
            val[sl] = 0
        for i in range(S):
            used_dev[i] = 0
            on_dev[i].clear()
        for t in range(T):
            used_srv[t] = 0
            row = used_house[t]
            for m in range(H):
                row[m] = 0
        total = 0
        for i in range(S):
            for c in range(C):
                for t, k in snap[i * C + c]:
                    apply_add(t, i, k)

    if relaxed_total > greedy_total:
        best_chain, best_total = relaxed_chain, relaxed_total
    else:
        best_chain, best_total = greedy_chain, greedy_total
    restore(best_chain)

    # ---- simulated annealing with reheat cycles ---------------------------
    # One long cooling run drifts away from good basins on the tighter seeds;
    # restarting from the incumbent every cycle keeps the search anchored.
    rng = random.Random(12345)
    # Scale the temperature to what a single move is actually worth. Moves add
    # or drop one rollout, so deltas are on the order of one rollout's value --
    # a temperature derived from the *total* instead leaves the search unable to
    # accept any intermediate loss, i.e. a hill climber that cannot escape a
    # basin needing several coordinated moves.
    n_roll = sum(len(chain[sl]) for sl in range(S * C))
    avg_roll = total / n_roll if n_roll else 1.0
    t_hot = max(1.0, avg_roll * 0.25)
    t_cold = max(0.001, avg_roll * 0.005)
    ratio = t_cold / t_hot
    all_slots = [(i, c) for i in range(S) for c in range(C)]
    n_slots = len(all_slots)
    exp = __import__("math").exp

    CYCLES = 12
    cycle_len = TIME_BUDGET / CYCLES
    for cycle in range(CYCLES):
        if cycle:
            restore(best_chain)
        cycle_start = time.time()
        if cycle_start - started >= TIME_BUDGET:
            break
        while True:
            frac = (time.time() - cycle_start) / cycle_len
            if frac >= 1.0 or time.time() - started >= TIME_BUDGET:
                break
            temp = t_hot * ratio ** frac
            for _ in range(300):
                roll = rng.random()

                if roll < 0.45:                                  # add
                    i = rng.randrange(S)
                    c = rng.randrange(C)
                    bucket = by_class[c]
                    if not bucket:
                        continue
                    k = bucket[int(rng.random() ** 2 * len(bucket))]
                    span = T - release[k]
                    if span <= 0:
                        continue
                    # Bias towards early days: they carry the (T - t) multiplier.
                    t = release[k] + int(rng.random() ** 2 * span)
                    if not can_add(t, i, k):
                        continue
                    delta = gain_of_add(t, i, k)
                    if delta > 0 or (delta / temp > -50.0
                                     and rng.random() < exp(delta / temp)):
                        apply_add(t, i, k)

                elif roll < 0.65:                                # remove
                    i, c = all_slots[rng.randrange(n_slots)]
                    lst = chain[i * C + c]
                    if not lst:
                        continue
                    t, k = lst[rng.randrange(len(lst))]
                    delta = slot_value(i, c, [x for x in lst if x != (t, k)]) - val[i * C + c]
                    if delta > 0 or (delta / temp > -50.0
                                     and rng.random() < exp(delta / temp)):
                        apply_remove(t, i, k)

                elif roll < 0.85:                                # move to another day
                    i, c = all_slots[rng.randrange(n_slots)]
                    lst = chain[i * C + c]
                    if not lst:
                        continue
                    t, k = lst[rng.randrange(len(lst))]
                    span = T - release[k]
                    if span <= 0:
                        continue
                    nt = release[k] + int(rng.random() ** 2 * span)
                    if nt == t:
                        continue
                    before = total
                    apply_remove(t, i, k)
                    if can_add(nt, i, k):
                        apply_add(nt, i, k)
                        delta = total - before
                        if delta <= 0 and not (delta / temp > -50.0
                                               and rng.random() < exp(delta / temp)):
                            apply_remove(nt, i, k)
                            apply_add(t, i, k)
                    else:
                        apply_add(t, i, k)

                else:                                            # swap in another update
                    i, c = all_slots[rng.randrange(n_slots)]
                    lst = chain[i * C + c]
                    if not lst:
                        continue
                    t, k = lst[rng.randrange(len(lst))]
                    bucket = by_class[c]
                    nk = bucket[int(rng.random() ** 2 * len(bucket))]
                    if nk == k:
                        continue
                    before = total
                    apply_remove(t, i, k)
                    if can_add(t, i, nk):
                        apply_add(t, i, nk)
                        delta = total - before
                        if delta <= 0 and not (delta / temp > -50.0
                                               and rng.random() < exp(delta / temp)):
                            apply_remove(t, i, nk)
                            apply_add(t, i, k)
                    else:
                        apply_add(t, i, k)

                if total > best_total:
                    best_chain, best_total = snapshot(), total

    restore(best_chain)
    while True:
        best_move, best_ratio = None, 0.0
        for t in range(T):
            for i in range(S):
                if used_dev[i] >= cap[i]:
                    continue
                for c in range(C):
                    if weight[i][c] == 0:
                        continue
                    for k in by_class[c]:
                        if not can_add(t, i, k):
                            continue
                        g = gain_of_add(t, i, k)
                        if g > 0 and g / size[k] > best_ratio:
                            best_ratio, best_move = g / size[k], (t, i, k)
        if best_move is None:
            break
        apply_add(*best_move)
    if total > best_total:
        best_chain, best_total = snapshot(), total

    plan = []
    for i in range(S):
        for c in range(C):
            for t, k in best_chain[i * C + c]:
                plan.append((t, i, k))

    out = [str(len(plan))]
    out.extend("%d %d %d" % row for row in plan)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
