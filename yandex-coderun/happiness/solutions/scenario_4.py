"""Scenario 4 - "Pyatnichnyi reliz": T=30, S=50, U=100, C=5, H=10, G=500.

Same objective identity as everywhere else: a (device, class) slot receiving
happiness h_1..h_p on days t_1 < ... < t_p contributes
w * sum_a h_a * (t_{a+1} - t_a) with t_{p+1} = T, so one rollout's marginal
value is w[i][c_k] * (h_k - h_on_slot) * (T - t).

What is new here is that the server budget actually binds: the house limits add
up to 750..1000 MB against G = 500, so on release days -- which cluster, this
being the Friday-release scenario -- houses compete for the server. The plan is
therefore built against both budgets at once rather than per house.

Scale forces two changes over the smaller scenarios. Candidates are restricted
to a per-class Pareto frontier: on a given day an update that is both bigger and
no happier than another available one can never be worth picking, which cuts
~20 updates per class down to 2..6. And the greedy runs off a heap holding one
best candidate per slot, instead of rescanning all S*C*U triples per rollout.
"""
import heapq
import random
import sys
import time

TIME_BUDGET = 3.0


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

    # ---- per-class Pareto frontier by day ---------------------------------
    # frontier[c][t]: updates of class c released by day t that no other such
    # update beats on both size and happiness. Kept sorted by size ascending,
    # so happiness is ascending too.
    released = [[] for _ in range(T)]
    for k in range(U):
        if release[k] < T:
            released[release[k]].append(k)
    frontier = [[None] * T for _ in range(C)]
    pool = [[] for _ in range(C)]
    for t in range(T):
        touched = set()
        for k in released[t]:
            pool[cls[k]].append(k)
            touched.add(cls[k])
        for c in range(C):
            if t and c not in touched:
                frontier[c][t] = frontier[c][t - 1]
                continue
            front, best_h = [], 0
            for k in sorted(pool[c], key=lambda k: (size[k], -happy[k])):
                if happy[k] > best_h:
                    best_h = happy[k]
                    front.append(k)
            frontier[c][t] = front

    # ---- state -------------------------------------------------------------
    chain = [[] for _ in range(S * C)]
    val = [0] * (S * C)
    used_dev = [0] * S
    on_dev = [set() for _ in range(S)]
    used_srv = [0] * T
    used_house = [[0] * H for _ in range(T)]
    total = 0

    def slot_value(i, c, lst):
        if not lst:
            return 0
        acc, last = 0, T
        for a in range(len(lst) - 1, -1, -1):
            t, k = lst[a]
            acc += happy[k] * (last - t)
            last = t
        return acc * weight[i][c]

    def can_add(t, i, k):
        if used_dev[i] >= cap[i] or t < release[k] or k in on_dev[i]:
            return False
        sz = size[k]
        if used_srv[t] + sz > G:
            return False
        m = house[i]
        if used_house[t][m] + sz > limit[m]:
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
        sl = i * C + cls[k]
        return slot_value(i, cls[k], sorted(chain[sl] + [(t, k)])) - val[sl]

    def held_at(i, c, t):
        """Happiness of the class-c model on device i at day t (0 if none)."""
        h = 0
        for dt, k in chain[i * C + c]:
            if dt <= t:
                h = happy[k]
            else:
                break
        return h

    def best_candidate(i, c, t):
        """Feasible update of class c maximising marginal gain per megabyte."""
        cur = held_at(i, c, t)
        best_k, best_ratio = -1, 0.0
        for k in frontier[c][t]:
            if happy[k] <= cur or k in on_dev[i] or not can_add(t, i, k):
                continue
            ratio = (happy[k] - cur) / size[k]
            if ratio > best_ratio:
                best_ratio, best_k = ratio, k
        if best_k < 0:
            return None
        return best_k, best_ratio * weight[i][c]

    def greedy_fill():
        """Fill remaining capacity, best marginal gain per megabyte first."""
        for t in range(T):
            heap = []
            for i in range(S):
                if used_dev[i] >= cap[i]:
                    continue
                base = i * C
                for c in range(C):
                    if weight[i][c] == 0:
                        continue
                    cand = best_candidate(i, c, t)
                    if cand:
                        heap.append((-cand[1], i, c, cand[0]))
            heapq.heapify(heap)
            while heap:
                _, i, c, k = heapq.heappop(heap)
                if used_dev[i] >= cap[i]:
                    continue
                if can_add(t, i, k):
                    apply_add(t, i, k)
                # A slot that just took a rollout is blocked for this class and
                # day, so best_candidate returns None and the loop terminates.
                cand = best_candidate(i, c, t)
                if cand:
                    heapq.heappush(heap, (-cand[1], i, c, cand[0]))

    def reset():
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

    def snapshot():
        return [list(x) for x in chain]

    def restore(snap):
        reset()
        for i in range(S):
            for c in range(C):
                for t, k in snap[i * C + c]:
                    apply_add(t, i, k)

    # ---- construction A: relaxation-guided --------------------------------
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
                    row = dp[q - 1]
                    for p in range(j):
                        if row[p] == NEG:
                            continue
                        cand = row[p] + (h - recs[p][1]) * (T - r)
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

    wanted = []
    for i in range(S):
        u = cap[i]
        row = [[0] * (u + 1) for _ in range(C + 1)]
        choice = [[0] * (u + 1) for _ in range(C + 1)]
        for c in range(C):
            prev = row[c]
            for used in range(u + 1):
                bv, bq = -1, 0
                for q in range(used + 1):
                    cand = prev[used - q] + weight[i][c] * F[c][q]
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
    greedy_fill()
    relaxed_chain, relaxed_total = snapshot(), total

    # ---- construction B: pure greedy ---------------------------------------
    reset()
    greedy_fill()
    greedy_chain, greedy_total = snapshot(), total

    if relaxed_total > greedy_total:
        best_chain, best_total = relaxed_chain, relaxed_total
    else:
        best_chain, best_total = greedy_chain, greedy_total
    restore(best_chain)

    # ---- simulated annealing ----------------------------------------------
    # Temperature tracks the value of one rollout, since every move adds, drops
    # or shifts exactly one; scaling it to the objective total instead would
    # make exp(-delta/T) vanish and reduce this to a hill climber.
    rng = random.Random(12345)
    n_roll = sum(len(x) for x in chain)
    avg_roll = total / n_roll if n_roll else 1.0
    t_hot = max(1.0, avg_roll * 0.10)
    t_cold = max(0.001, avg_roll * 0.005)
    ratio = t_cold / t_hot
    exp = __import__("math").exp
    all_slots = [(i, c) for i in range(S) for c in range(C)]
    n_slots = len(all_slots)

    CYCLES = 8
    cycle_len = TIME_BUDGET / CYCLES
    for cycle in range(CYCLES):
        if cycle:
            restore(best_chain)
        cycle_start = time.time()
        if cycle_start - started >= TIME_BUDGET:
            break
        while True:
            now = time.time()
            frac = (now - cycle_start) / cycle_len
            if frac >= 1.0 or now - started >= TIME_BUDGET:
                break
            temp = t_hot * ratio ** frac
            for _ in range(400):
                roll = rng.random()

                if roll < 0.45:                                  # add
                    i, c = all_slots[rng.randrange(n_slots)]
                    t = int(rng.random() ** 2 * T)
                    front = frontier[c][t]
                    if not front:
                        continue
                    k = front[rng.randrange(len(front))]
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
                    front = frontier[c][t]
                    if not front:
                        continue
                    nk = front[rng.randrange(len(front))]
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
    greedy_fill()
    if total > best_total:
        best_chain, best_total = snapshot(), total

    plan = []
    for i in range(S):
        base = i * C
        for c in range(C):
            for t, k in best_chain[base + c]:
                plan.append((t, i, k))

    out = [str(len(plan))]
    out.extend("%d %d %d" % row for row in plan)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
