"""Episode Exact Match scorer for the dev proxy."""
import json, sys
gold_path, pred_path = sys.argv[1], sys.argv[2]
gold = {}
for line in open(gold_path, encoding="utf-8"):
    e = json.loads(line)
    for q in e["queries"]:
        gold[(e["episode_id"], q["query_id"])] = q["label"]
pred, by_ep = {}, {}
for line in open(pred_path, encoding="utf-8"):
    r = json.loads(line)
    assert r["label"] == int(r["probability"] >= 0.5), r
    key = (r["episode_id"], r["query_id"])
    assert key in gold and key not in pred, key
    pred[key] = r["label"]
    by_ep.setdefault(r["episode_id"], []).append(r["label"] == gold[key])
assert len(pred) == len(gold), (len(pred), len(gold))
em = sum(all(v) for v in by_ep.values()) / len(by_ep)
acc = sum(sum(v) for v in by_ep.values()) / len(gold)
print(f"episodes={len(by_ep)} exact_match={em*100:.2f} query_accuracy={acc*100:.2f}")
