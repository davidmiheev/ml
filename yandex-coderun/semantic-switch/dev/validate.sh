#!/usr/bin/env bash
# Full local validation of the solution on the synthetic proxy corpora.
# Usage: dev/validate.sh   (run from the scratch root containing contest/ and dev/)
set -euo pipefail
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$(dirname "$0")/.."
mkdir -p contest/runs

run () {           # run <name> <train> <eval-nolabel> <eval-gold>
  local name=$1 train=$2 data=$3 gold=$4
  echo "=== $name"
  ( cd contest && python3 -m starter.solution.train --train "../$train" \
      --adapter "runs/$name.safetensors" --seed 1729 ) \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print('  chosen:',d['chosen'],'| train %.0fs'%d['total_seconds'],'| scalars',d['adapter_scalars'],'| bytes',d['adapter_bytes'])"
  ( cd contest && python3 -m starter.solution.predict --data "../$data" \
      --adapter "runs/$name.safetensors" --output "runs/${name}_pred.jsonl" ) \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print('  predict %.0fs'%d['total_seconds'])"
  printf '  '; python3 dev/score.py "$gold" "contest/runs/${name}_pred.jsonl"
}

run unseen_rules dev/data/train.jsonl        dev/data/test_nolabel.jsonl         dev/data/test.jsonl
run seen_rules   dev/data/heldin_train.jsonl dev/data/heldin_eval_nolabel.jsonl  dev/data/heldin_eval.jsonl
run free_balance dev/data_free/train.jsonl   dev/data_free/test_nolabel.jsonl    dev/data_free/test.jsonl

echo "=== contract test + packaging"
( cd contest && python3 -m pytest -q starter/tests/test_contract.py && \
  python3 package_submission.py --source starter/solution --output runs/submission.zip \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('  submission:',d['archive_bytes'],'bytes,',d['python_file_count'],'files')" )
echo "=== degenerate inputs"
python3 dev/edge_cases.py | tail -1
