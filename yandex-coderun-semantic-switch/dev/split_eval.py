"""Make a held-in evaluation split (rules seen in training) plus its train part."""
import json, sys, random
src, out_train, out_eval, n_eval = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
lines = [l for l in open(src, encoding="utf-8") if l.strip()]
random.Random(0).shuffle(lines)
with open(out_train, "w", encoding="utf-8") as f:
    f.writelines(lines[n_eval:])
with open(out_eval, "w", encoding="utf-8") as f:
    f.writelines(lines[:n_eval])
print(f"train={len(lines)-n_eval} eval={n_eval}")
