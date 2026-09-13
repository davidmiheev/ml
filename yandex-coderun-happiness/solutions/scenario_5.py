"""Scenario 5 - "The Last Dance": T=90, S=1000, U=500, C=10, H=100, G=5000.

Same marginal-value identity as the other scenarios: a rollout is worth
w[i][c_k] * (h_k - h_on_slot) * (T - t).

Two facts shape the algorithm here.

Bandwidth binds hard. The server hands out G*T = 450000 MB against an average
model of ~100 MB, so only ~4400 rollouts fit, while the device budgets ask for
~9900. Well under half the device slots can be used, which makes this a
"which rollouts are worth their megabytes" problem rather than a scheduling
one, and the greedy ranks by marginal gain per megabyte throughout.

Scale rules out the per-day rebuilds used in the smaller scenarios. S*C = 10000
slots over 90 days would mean 900k candidate evaluations per construction. Two
things avoid that:

* the (T - t) factor multiplies every candidate on a given day equally, so it
  does not affect their *ranking* -- the priority w[i][c] * (h_k - cur) / s_k is
  day-independent, and one heap can persist across all 90 days instead of being
  rebuilt per day;
* the best candidate for a slot depends only on (class, happiness currently
  held, house limit), not on which device it is, so it is memoised and shared
  across the thousand devices.

Also, with sizes up to 200 and house limits from 20, nearly half of all
(update, house) pairs can never be delivered even on a completely free day.
Those are rejected inside the candidate search rather than retried per day.

Best-solution tracking uses an undo journal rather than snapshots: copying
10000 slot chains on every improvement would cost more than the search itself.
"""
import heapq
import random
import sys
import time

TIME_BUDGET = 4.0


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

    min_size = min(size) if U else 1
    dev_limit = [limit[house[i]] for i in range(S)]

    # ---- per-class Pareto frontier by day, with a version per change -------
    released = [[] for _ in range(T)]
    for k in range(U):
        if release[k] < T:
            released[release[k]].append(k)
    frontier = [[None] * T for _ in range(C)]
    front_ver = [[0] * T for _ in range(C)]
    pool = [[] for _ in range(C)]
    ver = [0] * C
    for t in range(T):
        touched = set(cls[k] for k in released[t])
        for k in released[t]:
            pool[cls[k]].append(k)
        for c in range(C):
            if t and c not in touched:
                frontier[c][t] = frontier[c][t - 1]
                front_ver[c][t] = ver[c]
                continue
            front, best_h = [], 0
            for k in sorted(pool[c], key=lambda k: (size[k], -happy[k])):
                if happy[k] > best_h:
                    best_h = happy[k]
                    front.append(k)
            if t and front == frontier[c][t - 1]:
                frontier[c][t] = frontier[c][t - 1]
            else:
                frontier[c][t] = front
                ver[c] += 1
            front_ver[c][t] = ver[c]

    # ---- state -------------------------------------------------------------
    chain = [[] for _ in range(S * C)]
    val = [0] * (S * C)
    used_dev = [0] * S
    on_dev = [set() for _ in range(S)]
    used_srv = [0] * T
    used_house = [[0] * H for _ in range(T)]
    total = 0
    journal = []
    recording = [False]

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
        if recording[0]:
            journal.append((1, t, i, k))

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
        if recording[0]:
            journal.append((0, t, i, k))

    def rollback_to(mark):
        """Undo journalled moves back to a mark, without journalling the undo."""
        recording[0] = False
        while len(journal) > mark:
            was_add, t, i, k = journal.pop()
            if was_add:
                apply_remove(t, i, k)
            else:
                apply_add(t, i, k)
        recording[0] = True

    def gain_of_add(t, i, k):
        sl = i * C + cls[k]
        return slot_value(i, cls[k], sorted(chain[sl] + [(t, k)])) - val[sl]

    def held_at(i, c, t):
        h = 0
        for dt, k in chain[i * C + c]:
            if dt <= t:
                h = happy[k]
            else:
                break
        return h

    # Best (h - cur)/s over the day's frontier depends only on the class, what
    # the slot already holds and the house's limit -- never on which device --
    # so it is shared across all devices with the same triple.
    cache = {}
    MISSING = object()
    # Ranking key w*(h - cur)/(s + alpha). alpha = 0 ranks by gain per megabyte,
    # right while bandwidth is scarce; alpha -> infinity ranks by raw gain, right
    # while the device budget is scarce. Neither extreme wins here, so the value
    # is chosen per instance by trying a few and keeping the best plan.
    ALPHA = [0.0]

    def best_candidate(i, c, t):
        cur = held_at(i, c, t)
        lm = dev_limit[i]
        key = (c, front_ver[c][t], cur, lm)
        hit = cache.get(key, MISSING)
        if hit is MISSING:
            best_k, best_ratio = -1, 0.0
            for k in frontier[c][t]:
                if happy[k] <= cur or size[k] > lm:
                    continue
                ratio = (happy[k] - cur) / (size[k] + ALPHA[0])
                if ratio > best_ratio:
                    best_ratio, best_k = ratio, k
            hit = None if best_k < 0 else (best_k, best_ratio)
            cache[key] = hit
        if hit is None:
            return None
        k, ratio = hit
        if k in on_dev[i]:  # rare; fall back to a direct scan for this slot
            best_k, best_ratio = -1, 0.0
            for kk in frontier[c][t]:
                if happy[kk] <= cur or size[kk] > lm or kk in on_dev[i]:
                    continue
                r = (happy[kk] - cur) / (size[kk] + ALPHA[0])
                if r > best_ratio:
                    best_ratio, best_k = r, kk
            if best_k < 0:
                return None
            k, ratio = best_k, best_ratio
        return k, ratio * weight[i][c]

    slot_gen = [0] * (S * C)

    def greedy_fill():
        """Fill remaining capacity, best marginal gain per megabyte first.

        One heap serves all days: the (T - t) factor scales every candidate on a
        day equally, so it cannot change their order.
        """
        heap = []
        for i in range(S):
            if used_dev[i] >= cap[i]:
                continue
            base = i * C
            for c in range(C):
                if weight[i][c] == 0:
                    continue
                cand = best_candidate(i, c, 0)
                if cand:
                    sl = base + c
                    slot_gen[sl] += 1
                    heap.append((-cand[1], sl, cand[0], slot_gen[sl]))
        heapq.heapify(heap)

        for t in range(T):
            pending = []
            while heap:
                if used_srv[t] + min_size > G:
                    break
                negkey, sl, k, gen = heapq.heappop(heap)
                if gen != slot_gen[sl]:
                    continue                       # superseded entry
                i, c = divmod(sl, C)
                if used_dev[i] >= cap[i]:
                    continue                       # slot is finished for good
                if release[k] > t or happy[k] <= held_at(i, c, t) or k in on_dev[i]:
                    cand = best_candidate(i, c, t)  # frontier moved on
                    if cand:
                        slot_gen[sl] += 1
                        heapq.heappush(heap, (-cand[1], sl, cand[0], slot_gen[sl]))
                    continue
                sz = size[k]
                if used_srv[t] + sz > G or used_house[t][house[i]] + sz > limit[house[i]]:
                    pending.append((negkey, sl, k, gen))   # retry tomorrow
                    continue
                apply_add(t, i, k)
                cand = best_candidate(i, c, t)
                if cand:
                    slot_gen[sl] += 1
                    pending.append((-cand[1], sl, cand[0], slot_gen[sl]))
            for entry in pending:
                heapq.heappush(heap, entry)
            if t + 1 < T:
                # A class whose frontier just grew deserves fresh priorities.
                for c in range(C):
                    if front_ver[c][t + 1] == front_ver[c][t]:
                        continue
                    base_c = c
                    for i in range(S):
                        if used_dev[i] >= cap[i] or weight[i][base_c] == 0:
                            continue
                        sl = i * C + base_c
                        cand = best_candidate(i, base_c, t + 1)
                        if cand:
                            slot_gen[sl] += 1
                            heapq.heappush(heap, (-cand[1], sl, cand[0], slot_gen[sl]))

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
            base = i * C
            for c in range(C):
                for t, k in snap[base + c]:
                    apply_add(t, i, k)

    live = [i * C + c for i in range(S) for c in range(C) if weight[i][c]]
    n_live = len(live)

    def upgrade_pass():
        """Spend leftover bandwidth raising the happiness of rollouts already placed.

        The greedy ranks by gain per (size + alpha), so it prefers small models
        and tends to leave the server well short of its cap. Swapping update k
        for a happier k' on the same day changes the objective by exactly
        w * (h' - h) * (days until the slot's next rollout), which is increasing
        in h', so the best swap is simply the happiest model that fits the room
        freed by dropping k. The frontier is ordered by size (hence by happiness
        too), so that is a backwards scan ending at the first model that fits.
        """
        changed = 0
        for sl in live:
            lst = chain[sl]
            if not lst:
                continue
            i, c = divmod(sl, C)
            m = house[i]
            lmt = limit[m]
            for idx in range(len(lst)):
                t, k = lst[idx]
                sz = size[k]
                room = G - used_srv[t] + sz
                room_h = lmt - used_house[t][m] + sz
                if room_h < room:
                    room = room_h
                front = frontier[c][t]
                pick = -1
                for j in range(len(front) - 1, -1, -1):
                    k2 = front[j]
                    if size[k2] > room:
                        continue
                    if happy[k2] <= happy[k]:
                        break          # frontier happiness ascends with size
                    if k2 not in on_dev[i]:
                        pick = k2
                        break
                if pick >= 0:
                    apply_remove(t, i, k)
                    apply_add(t, i, pick)
                    changed += 1
        return changed

    # ---- construction: sweep the ranking parameter ------------------------
    mean_size = (sum(size) / U) if U else 1.0
    best_chain, best_build = None, -1
    for factor in (0.0, 0.25, 0.5, 1.0, 2.0):
        ALPHA[0] = factor * mean_size
        reset()
        cache.clear()
        greedy_fill()
        for _ in range(3):
            if not upgrade_pass():
                break
        greedy_fill()
        if total > best_build:
            best_build, best_chain = total, snapshot()
            best_alpha = ALPHA[0]
        if time.time() - started > TIME_BUDGET * 0.5:
            break
    ALPHA[0] = best_alpha
    cache.clear()
    restore(best_chain)
    build_done = time.time()

    # ---- annealing on top, with the journal marking the incumbent ----------
    rng = random.Random(12345)
    n_roll = sum(len(x) for x in chain)
    avg_roll = total / n_roll if n_roll else 1.0
    t_hot = max(1.0, avg_roll * 0.10)
    t_cold = max(0.001, avg_roll * 0.005)
    ratio_tc = t_cold / t_hot
    exp = __import__("math").exp
    best_total = total
    best_mark = 0
    recording[0] = True

    CYCLES = 6
    remaining = TIME_BUDGET - (build_done - started)
    cycle_len = remaining / CYCLES if remaining > 0 else 0.0
    for cycle in range(CYCLES):
        if cycle:
            rollback_to(best_mark)
            journal.clear()
            best_mark = 0
        cycle_start = time.time()
        if cycle_start - started >= TIME_BUDGET:
            break
        while True:
            now = time.time()
            frac = (now - cycle_start) / cycle_len if cycle_len else 1.0
            if frac >= 1.0 or now - started >= TIME_BUDGET:
                break
            temp = t_hot * ratio_tc ** frac
            for _ in range(300):
                sl = live[rng.randrange(n_live)]
                i, c = divmod(sl, C)
                lst = chain[sl]
                roll = rng.random()

                if roll < 0.25 or not lst:                       # add
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

                elif roll < 0.40:                                # remove
                    t, k = lst[rng.randrange(len(lst))]
                    delta = slot_value(i, c, [x for x in lst if x != (t, k)]) - val[sl]
                    if delta > 0 or (delta / temp > -50.0
                                     and rng.random() < exp(delta / temp)):
                        apply_remove(t, i, k)

                elif roll < 0.75:                                # move to another day
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

                else:                                            # swap update
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
                    best_total = total
                    best_mark = len(journal)

    rollback_to(best_mark)
    recording[0] = False
    cache.clear()
    greedy_fill()

    plan = []
    for i in range(S):
        base = i * C
        for c in range(C):
            for t, k in chain[base + c]:
                plan.append((t, i, k))

    out = [str(len(plan))]
    out.extend("%d %d %d" % row for row in plan)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
