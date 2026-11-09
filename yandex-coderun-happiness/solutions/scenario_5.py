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

The construction leaves every day full until the device budgets run out
(~day 60), after which the server idles: bandwidth and budget bind at once.
No single-rollout move can improve such a schedule -- adding needs room,
moving later loses value, a happier model needs room -- and annealing over
those moves found nothing in a minute. The search is therefore ruin & recreate:
free the rollouts of a few devices, a house or a window of days, refill with
the greedy under jittered ranking parameters, keep the result if no worse.
Rejected attempts are undone through a journal rather than snapshots.
"""
import heapq
import os
import random
import sys
import time

TIME_BUDGET = float(os.environ.get("TIME_BUDGET", 20.0))
BETA_FACTOR = 0.25   # ranking uses w*(h-cur)/(s + BETA_FACTOR*mean size)


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
    # Two parameters. The slot's model is the frontier item maximising
    # (h - cur)/(s + alpha): alpha = 0 is gain per megabyte, right while
    # bandwidth is scarce; alpha -> infinity is raw gain, right while the device
    # budget is scarce. Both bind here, so alpha is swept per instance. Once the
    # model is fixed, slots are *ordered* by w*(h - cur)/(s + beta) with a
    # smaller beta: with the selection made, sequencing is closer to a
    # weighted-completion-time problem, where the ratio rule is optimal.
    ALPHA = [0.0]
    BETA = [0.0]

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
            hit = None if best_k < 0 else (best_k, (happy[best_k] - cur) / (size[best_k] + BETA[0]))
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
            k, ratio = best_k, (happy[best_k] - cur) / (size[best_k] + BETA[0])
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
    BETA[0] = BETA_FACTOR * mean_size
    for factor in (0.5, 1.0, 1.5, 2.0):
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
            best_beta = BETA[0]
        if time.time() - started > TIME_BUDGET * 0.5:
            break
    ALPHA[0] = best_alpha
    BETA[0] = best_beta
    cache.clear()
    restore(best_chain)

    # ---- ruin & recreate on top ---------------------------------------------
    # Single-rollout moves cannot improve a saturated schedule (every day is
    # full until the budgets run out), so the search frees a random region --
    # the rollouts of a few devices, of one house, or of a window of days --
    # and lets the greedy refill it, keeping the result when it is no worse.
    # The refill runs with alpha and beta jittered around the swept optimum;
    # without that the greedy mostly rebuilds what it tore down.
    rng = random.Random(12345)
    recording[0] = True
    best_total = total
    n_iter = n_acc = 0
    dev_rolls = lambda i: [(t, k) for c in range(C) for t, k in chain[i * C + c]]
    by_house = [[] for _ in range(H)]
    for i in range(S):
        by_house[house[i]].append(i)
    while time.time() - started < TIME_BUDGET:
        n_iter += 1
        mark = len(journal)
        roll = rng.random()
        if roll < 0.45:
            victims = [rng.randrange(S) for _ in range(rng.randint(10, 40))]
        elif roll < 0.7:
            victims = list(by_house[rng.randrange(H)])
            victims.extend(rng.randrange(S) for _ in range(5))
        else:
            t0 = rng.randrange(T)
            days = set(range(t0, min(T, t0 + rng.randint(1, 3))))
            victims = []
            for i in range(S):
                for t, k in dev_rolls(i):
                    if t in days:
                        apply_remove(t, i, k)
        for i in victims:
            for t, k in dev_rolls(i):
                apply_remove(t, i, k)
        ALPHA[0] = best_alpha * rng.uniform(0.6, 1.6)
        BETA[0] = best_beta * rng.uniform(0.6, 1.6)
        cache.clear()
        greedy_fill()
        upgrade_pass()
        if total >= best_total:
            if total > best_total:
                n_acc += 1
            best_total = total
            journal.clear()
        else:
            rollback_to(mark)
    sys.stderr.write('build %d -> lns %d after %d iters, %d improvements\n' % (best_build, best_total, n_iter, n_acc))
    recording[0] = False
    ALPHA[0], BETA[0] = best_alpha, best_beta
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
