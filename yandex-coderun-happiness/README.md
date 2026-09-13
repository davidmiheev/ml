# Счастье — Yandex CodeRun ML contest

Scheduling model rollouts to smart devices over `T` days to maximise total
weighted user happiness. Full statement in [`statement/`](statement/).

## The core identity

The objective is stated as a sum over every day, device and class of
`w[i][j] * p[i][j][t]`, where `p` is the happiness of the model sitting in that
slot. That telescopes: if a (device, class) slot receives updates with
happiness `h_1..h_p` on days `t_1 < ... < t_p`, its contribution is

    w * sum_a h_a * (t_{a+1} - t_a)          (t_{p+1} = T)
  = w * sum_a (h_a - h_{a-1}) * (T - t_a)    (h_0 = 0)

so every individual rollout carries an independent marginal value

    gain(t, i, k) = w[i][c_k] * (h_k - h_currently_on_slot) * (T - t)

which is what all of the solutions here optimise against. Three consequences:

* a rollout only pays off when `h_k` beats what the slot already holds, so each
  slot's chain of updates strictly increases in happiness;
* bandwidth is per-day and does not carry over, so unused bandwidth is lost —
  fill every day;
* the device budget `u_i` is the only *global* resource, so spending a slot on a
  mediocre model early can lock out a much better one later. That is where pure
  day-by-day greed fails and lookahead is needed.

## Layout

    solutions/     one self-contained stdlib-only file per scenario
    statement/     problem statement and the per-scenario test descriptions
    tools/gen.py   random tests matching each scenario's stated constraints
    tools/score.py validator + scorer (reproduces the statement's worked example)
    tools/run.py   generate, solve and score a scenario across seeds
    tools/brute2.py exhaustive optimum for scenario 2, certifies that solution
    tools/bound.py  infinite-bandwidth upper bound, to measure how much is left
    tools/compare.py score a scenario against that bound across seeds

Solutions use only the standard library, since the judge's environment is
unknown and the provided templates are plain Python.

    python3 tools/run.py 3 0 1 2      # solve scenario 3 on seeds 0,1,2

## Reading the numbers

`tools/bound.py` drops both bandwidth limits, so it is only tight when the
device budget is what binds. On instances where the generator puts every device
in one house, a single 50 MB/day pipe serves the whole fleet and the plan
saturates it -- there the solution sits at ~68% of the bound while being within
a fraction of a percent of what a 20x longer search finds. Judge a scenario by
the gap to a long run, and use the bound only to spot device-budget-bound cases.

## Status

| # | Scenario | Points | Approach | State |
|---|----------|--------|----------|-------|
| 1 | Проверка связи | — | single forced rollout, feasibility-checked | optimal |
| 2 | CDN на коленке | 5 | one rollout/day, devices by weight into earliest days | optimal |
| 3 | Две недели без доступов | 10 | relaxation-guided + greedy seeds, then annealing | ~99% of a 22x longer run |
| 4 | Пятничный релиз | 20 | | |
| 5 | The Last Dance | 25 | | |
| 6 | Наследие | 40 | | |
