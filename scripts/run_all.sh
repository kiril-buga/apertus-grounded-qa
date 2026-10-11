#!/usr/bin/env bash
# Full pipeline: data -> baseline -> SFT -> DPO -> verify -> report.
# Usage: bash scripts/run_all.sh configs/apertus8b.yaml [runs]
set -euo pipefail

CFG=${1:-configs/apertus8b.yaml}
RUNS=${2:-runs}
SFT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['sft']['output_dir'])" "$CFG")
DPO=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['dpo']['output_dir'])" "$CFG")
DATA=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['dir'])" "$CFG")
K=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['prefs']['samples_per_prompt'])" "$CFG")
T=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['prefs']['temperature'])" "$CFG")

step() { echo; echo "=== $* ==="; }

step "1/8 prepare data"
python -m groundqa.prepare --config "$CFG"

step "2/8 baseline answers"
python -m groundqa.generate --config "$CFG" --out "$RUNS/preds/base.jsonl"

step "3/8 SFT"
python -m groundqa.train_sft --config "$CFG"

step "4/8 SFT answers"
python -m groundqa.generate --config "$CFG" --adapter "$SFT" --out "$RUNS/preds/sft.jsonl"

step "5/8 sample the SFT model on held-out prompts and build preference pairs"
python -m groundqa.generate --config "$CFG" --adapter "$SFT" --split pref \
  --num-samples "$K" --temperature "$T" --out "$RUNS/preds/sft_pref_samples.jsonl"
python -m groundqa.prefs --config "$CFG" --split pref \
  --samples "$RUNS/preds/sft_pref_samples.jsonl" --out "$DATA/prefs.jsonl"

step "6/8 DPO"
python -m groundqa.train_dpo --config "$CFG" --sft-adapter "$SFT" --prefs "$DATA/prefs.jsonl"

step "7/8 SFT+DPO answers"
python -m groundqa.generate --config "$CFG" --adapter "$SFT" --adapter "$DPO" --out "$RUNS/preds/dpo.jsonl"

step "8/8 verify and report"
for run in base sft dpo; do
  python -m groundqa.verify --config "$CFG" --preds "$RUNS/preds/$run.jsonl" --out "$RUNS/verified/$run.jsonl"
done
python -m groundqa.report \
  --run base="$RUNS/verified/base.jsonl" \
  --run sft="$RUNS/verified/sft.jsonl" \
  --run sft+dpo="$RUNS/verified/dpo.jsonl" \
  --out results/results.md

echo
echo "Next: python -m groundqa.annotate export --config $CFG --verified base=$RUNS/verified/base.jsonl --verified sft=$RUNS/verified/sft.jsonl --verified dpo=$RUNS/verified/dpo.jsonl"
