"""Scenario 6 - "Наследие": T=365, S=100000, U=10000, C=30, H=10000, G=10^6.

Same marginal-value identity as before: a rollout is worth
w[i][c_k] * (h_k - h_on_slot) * (T - t), so the day-independent priority
w * (h - cur) / (s + alpha) from scenario 5 carries over, and so does the
persistent heap.  What does not carry over is one heap entry per slot: there
are S*C = 3 million slots, and a class's frontier changes ~17 times over the
year, each change re-ranking every slot of that class.

The saving observation is that the priority factorises.  For a slot the best
candidate and its (h - cur)/(s + alpha) depend only on (class, happiness held,
house limit) -- the "group" of the slot -- and the slot's own contribution is
the single factor w.  Inside a group, slots are therefore ranked by w alone,
which never changes.  So each group keeps its slots in a list sorted by w, and
the heap holds one entry per group keyed by w_head * ratio.  That is ~30000
groups for untouched slots plus one per (class, held happiness, limit) that
actually occurs, instead of 3 million, and a frontier change re-keys a few
thousand groups instead of 100000.

Everything else is bookkeeping to keep the per-group work O(1) per pop:
slots whose house is full today are parked and restored at the end of the
day, slots whose device ran out of budget are dropped when they reach the head,
and a group whose current candidate does not fit the server's remaining room
today is deferred to tomorrow in one step, since that candidate is the same for
every slot in it.

Ranking by gain per (s + alpha) rather than per megabyte: bandwidth is the
scarce resource here (~730k mean-sized rollouts fit against ~1.75M device
slots), but small models are so much cheaper that pure gain-per-megabyte
fills the device budgets with tiny mediocre models.  alpha is a fraction of
the mean model size, chosen from the one construction there is time for.

After the greedy, an upgrade pass spends leftover room on each day by swapping
a placed model for the happiest one on that day's frontier that fits: that
changes the objective by exactly w * (h' - h) * (days until the slot's next
rollout) > 0, so it never needs to be evaluated, only checked for room.
"""
import heapq
import sys
import time
from bisect import bisect_right, insort

ALPHA_FACTOR = 2.0    # alpha as a fraction of mean model size
BETA = 0.0            # alpha scales with (w/50)^beta when choosing a slot's model
MAX_SERVER_FAILS = int(__import__("os").environ.get("FAILS", 40))  # consecutive candidates too big for today's room -> day over


def main():
    started = time.time()
    log = lambda msg: sys.stderr.write("%6.2fs %s\n" % (time.time() - started, msg))
    data = sys.stdin.buffer.read().split()
    T, S, U, C, H, G = (int(x) for x in data[:6])
    pos = 6
    limit = list(map(int, data[pos:pos + H]))
    pos += H
    stride = C + 2
    dev = list(map(int, data[pos:pos + S * stride]))
    pos += S * stride
    house = dev[0::stride]
    cap = dev[1::stride]
    upd = list(map(int, data[pos:pos + 4 * U]))
    cls = upd[0::4]
    size = upd[1::4]
    happy = upd[2::4]
    release = upd[3::4]

    log("parsed")
    mean_size = (sum(size) / U) if U else 1.0
    import os
    alpha = float(os.environ.get("ALPHA", ALPHA_FACTOR)) * mean_size
    beta = float(os.environ.get("BETA", BETA))
    max_h = max(happy) if U else 0
    # weight buckets: the model chosen for a slot depends on its weight
    # through w*(h-cur) - lam*s, so slots are grouped by a coarse weight class
    # experimental: NB=8 groups slots by weight bucket too (so far ~2.7% worse)
    edges = [1, 2, 4, 7, 12, 20, 33, 55, 101] if os.environ.get("NB") == "8" else [1, 101]
    bucket_of = [0] * 101
    bucket_w = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1] - 1
        bucket_w.append((lo * hi) ** 0.5)
        for w in range(lo, hi + 1):
            bucket_of[w] = b

    # ---- per-class Pareto frontier, versioned -----------------------------
    # fronts[c][v]: updates of class c on the frontier under version v, sorted
    # by size (and hence strictly ascending in happiness); fver[c][t]: version
    # in force on day t.
    released = [[] for _ in range(T)]
    for k in range(U):
        if release[k] < T:
            released[release[k]].append(k)
    fronts = [[[]] for _ in range(C)]
    fsizes = [[[]] for _ in range(C)]
    fver = [[0] * T for _ in range(C)]
    pool = [[] for _ in range(C)]
    for t in range(T):
        touched = set()
        for k in released[t]:
            pool[cls[k]].append(k)
            touched.add(cls[k])
        for c in range(C):
            v = fver[c][t - 1] if t else 0
            if c in touched:
                front, best_h = [], 0
                for k in sorted(pool[c], key=lambda k: (size[k], -happy[k])):
                    if happy[k] > best_h:
                        best_h = happy[k]
                        front.append(k)
                if front != fronts[c][v]:
                    fronts[c].append(front)
                    fsizes[c].append([size[k] for k in front])
                    v = len(fronts[c]) - 1
            fver[c][t] = v

    # ---- state -------------------------------------------------------------
    used_srv = [0] * T
    used_house = [0] * (T * H)
    slot_day = [-1] * (S * C)   # day of the slot's latest rollout
    roll_t, roll_i, roll_k = [], [], []
    rolled = set()      # sl * U + k for every rollout made

    # ---- which classes each device spends its budget on ---------------------
    # A device has fewer rollouts than classes, so grabbing whatever class has
    # a decent model today spends budget the device would rather keep for a
    # class it values more whose good model comes out next week. Each device
    # therefore commits its budget to the classes with the best
    # w * max_k h_k (T - r_k) / (s_k + alpha) over the whole horizon (models
    # that fit its house), and only budget beyond that is spent on upgrades.
    max_lm = max(limit) if H else 0
    class_val = []
    for c in range(C):
        arr = [0.0] * (max_lm + 1)
        for k in pool[c]:
            s = size[k]
            if s <= max_lm:
                val = happy[k] * (T - release[k]) / (s + alpha)
                if val > arr[s]:
                    arr[s] = val
        for s in range(1, max_lm + 1):
            if arr[s - 1] > arr[s]:
                arr[s] = arr[s - 1]
        class_val.append(arr)
    spare = [0] * S     # rollouts left for upgrades once planned slots are done

    # rows[(c, v, cur)][p]: (best k, (h - cur)/(s + alpha)) over the first p
    # frontier items with h > cur, or None.
    rows = {}

    def get_row(c, v, cur, b):
        key = (c, v, cur, b)
        row = rows.get(key)
        if row is None:
            front = fronts[c][v]
            ab = alpha * (bucket_w[b] / 50.0) ** beta
            row = [None]
            best_k, best_net, best_r = -1, 0.0, 0.0
            for k in front:
                if happy[k] > cur:
                    net = (happy[k] - cur) / (size[k] + ab)
                    if net > best_net:
                        best_net, best_k = net, k
                        best_r = (happy[k] - cur) / (size[k] + alpha)
                row.append((best_k, best_r) if best_k >= 0 else None)
            rows[key] = row
        return row

    # ---- groups ------------------------------------------------------------
    g_list, g_c, g_cur, g_lm, g_b, g_gen, g_parked = [], [], [], [], [], [], []
    gid_of = {}
    class_groups = [[] for _ in range(C)]

    def new_group(c, cur, lm, b):
        gid = len(g_list)
        g_list.append([])
        g_c.append(c)
        g_cur.append(cur)
        g_lm.append(lm)
        g_b.append(b)
        g_gen.append(0)
        g_parked.append([])
        gid_of[(c, cur, lm, b)] = gid
        class_groups[c].append(gid)
        return gid

    nlargest = heapq.nlargest
    for i in range(S):
        lm = limit[house[i]]
        base = i * stride + 2
        u = cap[i]
        keyed = [(dev[base + c] * class_val[c][lm], c)
                 for c in range(C) if dev[base + c]]
        if len(keyed) > u:
            keyed = nlargest(u, keyed)
        spare[i] = u - len(keyed)
        sl = i * C
        for _, c in keyed:
            w = dev[base + c]
            b = bucket_of[w]
            gid = gid_of.get((c, 0, lm, b))
            if gid is None:
                gid = new_group(c, 0, lm, b)
            g_list[gid].append((w, sl + c))
    tiekey = (lambda e: (e[0], -e[1])) if os.environ.get('TIE') else None
    for lst in g_list:
        lst.sort(key=tiekey)
    log("groups built: %d" % len(g_list))

    heap = []

    def push_group(gid, v):
        lst = g_list[gid]
        if not lst:
            return
        c = g_c[gid]
        row = get_row(c, v, g_cur[gid], g_b[gid])
        ent = row[bisect_right(fsizes[c][v], g_lm[gid])]
        if ent is None:
            return
        g_gen[gid] += 1
        heap.append((-lst[-1][0] * ent[1], gid, g_gen[gid]))

    for gid in range(len(g_list)):
        push_group(gid, fver[g_c[gid]][0])
    heapq.heapify(heap)
    log("heap ready")
    n_pop = n_park = n_dead = n_fail = n_stale = 0

    # ---- the greedy --------------------------------------------------------
    heappop, heappush = heapq.heappop, heapq.heappush
    for t in range(T):
        room = G - used_srv[t]
        pending = []
        touched = []
        fails = 0
        hbase = t * H
        while heap and fails < MAX_SERVER_FAILS:
            negkey, gid, gen = heappop(heap)
            n_pop += 1
            if gen != g_gen[gid]:
                continue
            lst = g_list[gid]
            c = g_c[gid]
            v = fver[c][t]
            row = get_row(c, v, g_cur[gid], g_b[gid])
            ent = row[bisect_right(fsizes[c][v], g_lm[gid])]
            if ent is None:
                continue
            k, ratio = ent
            s = size[k]
            if s > room:
                fails += 1
                n_fail += 1
                pending.append(gid)
                continue
            upgrade = g_cur[gid] > 0
            while lst:
                w, sl = lst[-1]
                i = sl // C
                if upgrade and spare[i] <= 0:
                    lst.pop()
                    n_dead += 1
                    continue
                key = w * ratio
                if key < -negkey - 1e-9 and heap and -heap[0][0] > key:
                    n_stale += 1
                    g_gen[gid] += 1
                    heappush(heap, (-key, gid, g_gen[gid]))
                    break
                m = house[i]
                if used_house[hbase + m] + s > limit[m] or slot_day[sl] == t:
                    lst.pop()
                    n_park += 1
                    if not g_parked[gid]:
                        touched.append(gid)
                    g_parked[gid].append((w, sl))
                    continue
                # take it
                lst.pop()
                fails = 0
                room -= s
                used_house[hbase + m] += s
                if upgrade:
                    spare[i] -= 1
                h = happy[k]
                slot_day[sl] = t
                roll_t.append(t)
                roll_i.append(i)
                roll_k.append(k)
                rolled.add(sl * U + k)
                if h < max_h and spare[i] > 0:
                    lm = g_lm[gid]
                    b = g_b[gid]
                    ng = gid_of.get((c, h, lm, b))
                    if ng is None:
                        ng = new_group(c, h, lm, b)
                    nl = g_list[ng]
                    if not nl or w >= nl[-1][0]:
                        nl.append((w, sl))
                        push_group(ng, v)
                    else:
                        insort(nl, (w, sl))
                if lst:
                    g_gen[gid] += 1
                    heappush(heap, (-lst[-1][0] * ratio, gid, g_gen[gid]))
                break
        used_srv[t] = G - room
        if t + 1 >= T:
            break
        repush = set(pending)
        for gid in touched:
            lst = g_list[gid]
            lst.extend(g_parked[gid])
            lst.sort(key=tiekey)
            g_parked[gid].clear()
            repush.add(gid)
        for c in range(C):
            if fver[c][t + 1] != fver[c][t]:
                repush.update(class_groups[c])
        for gid in repush:
            push_group(gid, fver[g_c[gid]][t + 1])

    log("greedy done: %d rollouts, pops %d park %d dead %d fail %d stale %d, groups %d, rows %d, srv used %.1f%%"
        % (len(roll_t), n_pop, n_park, n_dead, n_fail, n_stale, len(g_list), len(rows),
           100.0 * sum(used_srv) / (G * T)))
    # ---- upgrade pass: spend leftover room on happier models ----------------
    n_roll = len(roll_t)
    n_up = 0
    for idx in range(n_roll):
        t, i, k = roll_t[idx], roll_i[idx], roll_k[idx]
        s = size[k]
        m = house[i]
        room = G - used_srv[t] + s
        room_h = limit[m] - used_house[t * H + m] + s
        if room_h < room:
            room = room_h
        if room <= s:
            continue
        c = cls[k]
        v = fver[c][t]
        front = fronts[c][v]
        h = happy[k]
        sl = i * C + c
        pick = -1
        for j in range(len(front) - 1, -1, -1):
            k2 = front[j]
            if size[k2] > room:
                continue
            if happy[k2] <= h:
                break
            if sl * U + k2 in rolled:
                continue
            pick = k2
            break
        if pick >= 0:
            s2 = size[pick]
            used_srv[t] += s2 - s
            used_house[t * H + m] += s2 - s
            roll_k[idx] = pick
            rolled.add(sl * U + pick)
            n_up += 1

    log("upgrade done: %d swaps, srv used %.1f%%, mean h %.1f, mean s %.1f"
        % (n_up, 100.0 * sum(used_srv) / (G * T),
           sum(happy[k] for k in roll_k) / max(1, n_roll),
           sum(size[k] for k in roll_k) / max(1, n_roll)))
    out = [str(n_roll)]
    out.extend("%d %d %d" % (roll_t[j], roll_i[j], roll_k[j]) for j in range(n_roll))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
