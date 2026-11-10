"""Degenerate-input checks for the solution pipeline."""
import json, subprocess, sys, tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "contest"
out = Path(tempfile.mkdtemp())


def episode(i, texts_sup, texts_qry, labeled=True):
    e = {"episode_id": f"e{i}",
         "support": [{"example_id": f"s{i}_{j}", "text": t, "label": 1 if j < 4 else 0}
                     for j, t in enumerate(texts_sup)],
         "queries": [{"query_id": f"q{i}_{j}", "text": t, **({"label": j % 2} if labeled else {})}
                     for j, t in enumerate(texts_qry)]}
    return e


def write(path, episodes):
    with open(path, "w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


cases = {}
same = ["Одинаковое сообщение."] * 8
cases["identical_support"] = [episode(i, same, ["Запрос один.", "Запрос два.", "Запрос три.", "Запрос четыре."])
                              for i in range(6)]
longtext = ["слово " * 200 + str(j) for j in range(8)]
cases["very_long"] = [episode(i, longtext, [("текст " * 150) + str(j) for j in range(4)]) for i in range(6)]
cases["tiny"] = [episode(0, [f"Поддержка {j}" for j in range(8)], [f"Запрос {j}" for j in range(4)])]
cases["weird_unicode"] = [episode(i, [f"🙂 ЁЖ {j} ñ 漢字   test" for j in range(8)],
                                  [f"emoji 🚀 {j}" for j in range(4)]) for i in range(6)]
cases["duplicate_queries"] = [episode(i, [f"S{j} текст" for j in range(8)], ["одно и то же"] * 4)
                              for i in range(6)]

fail = 0
for name, eps in cases.items():
    train_path, eval_path = out / f"{name}_train.jsonl", out / f"{name}_eval.jsonl"
    write(train_path, eps)
    write(eval_path, [episode(i, e["support"] and [x["text"] for x in e["support"]],
                              [q["text"] for q in e["queries"]], labeled=False)
                      for i, e in enumerate(eps)])
    adapter, preds = out / f"{name}.safetensors", out / f"{name}_pred.jsonl"
    for cmd in (
        [sys.executable, "-m", "starter.solution.train", "--train", str(train_path),
         "--adapter", str(adapter), "--epochs", "1", "--batch-size", "2", "--seed", "7"],
        [sys.executable, "-m", "starter.solution.predict", "--data", str(eval_path),
         "--adapter", str(adapter), "--output", str(preds), "--batch-size", "2"],
    ):
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "TRANSFORMERS_OFFLINE": "1",
                                "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(root)})
        if r.returncode != 0:
            print(f"FAIL {name}: {cmd[2]}\n{r.stderr[-1500:]}")
            fail += 1
            break
    else:
        rows = [json.loads(l) for l in open(preds, encoding="utf-8")]
        bad = [r for r in rows if r["label"] != int(r["probability"] >= 0.5)
               or not (0.0 <= r["probability"] <= 1.0)]
        seen = {(r["episode_id"], r["query_id"]) for r in rows}
        ok = not bad and len(seen) == len(rows) == 4 * len(eps)
        print(f"{'OK  ' if ok else 'BAD '} {name}: {len(rows)} rows, contract={'ok' if ok else 'violated'}")
        fail += 0 if ok else 1
print("edge cases failed:", fail)
sys.exit(1 if fail else 0)
