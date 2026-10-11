"""Check the automatic metrics against human labels before trusting them.

1. Export a stratified sample to a CSV and label it by hand:
    python -m groundqa.annotate export --config configs/apertus8b.yaml \
        --verified base=runs/verified/base.jsonl --verified sft=runs/verified/sft.jsonl \
        --n 100 --out results/annotation.csv

   Fill in, per row:
     human_abstained       y/n   did the model decline to answer?
     human_unsupported     claim numbers not supported by the context, e.g. "2,3" (blank = all supported)
     human_correct         y/n   is this the right behaviour overall (correct grounded answer, or a justified abstention)?

2. Measure agreement (Cohen's kappa) of the verifier, and of an LLM judge if
   judge.py filled the judge_correct column, with your labels:
    python -m groundqa.annotate agree --csv results/annotation.csv
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from .config import load_config, read_jsonl
from .metrics import cohen_kappa, is_grounded_correct

FIELDS = [
    "run", "id", "question", "context", "reference", "prediction", "claims",
    "verifier_abstained", "verifier_unsupported", "verifier_correct",
    "judge_correct", "human_abstained", "human_unsupported", "human_correct", "notes",
]


def export(args) -> None:
    cfg = load_config(args.config)
    examples = {ex["id"]: ex for ex in read_jsonl(Path(cfg["data"]["dir"]) / f"{args.split}.jsonl")}
    rng = random.Random(args.seed)
    out_rows = []
    for spec in args.verified:
        run, _, path = spec.partition("=")
        rows = [r for r in read_jsonl(path) if r.get("sample", 0) == 0]
        # Stratify by (answerable, abstained) so rare cells such as wrong abstentions are represented.
        strata = defaultdict(list)
        for r in rows:
            strata[(r["answerable"], r["abstained"])].append(r)
        per = max(1, args.n // max(1, len(strata)))
        picked = []
        for cell in strata.values():
            picked += rng.sample(cell, min(per, len(cell)))
        for r in picked:
            ex = examples[r["id"]]
            out_rows.append({
                "run": run,
                "id": r["id"],
                "question": ex["question"],
                "context": ex["context"],
                "reference": ex["answer"] or "(unanswerable)",
                "prediction": r["prediction"],
                "claims": "\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(r["claims"])),
                "verifier_abstained": "y" if r["abstained"] else "n",
                "verifier_unsupported": ",".join(str(i + 1) for i, c in enumerate(r["claims"]) if not c["supported"]),
                "verifier_correct": "y" if is_grounded_correct(r) else "n",
                "judge_correct": "", "human_abstained": "", "human_unsupported": "", "human_correct": "", "notes": "",
            })
    rng.shuffle(out_rows)  # hide which run produced which row while labelling
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {args.out}. Label the human_* columns, then run `agree`.")


def _claim_labels(row: dict, column: str) -> list[bool]:
    n = row["claims"].count("\n") + 1 if row["claims"].strip() else 0
    bad = {int(x) for x in row[column].replace(" ", "").split(",") if x}
    return [i + 1 not in bad for i in range(n)]


def agreement(rows: list[dict]) -> dict[str, dict]:
    labelled = [r for r in rows if r["human_correct"].strip()]
    out = {}

    def pair(name, a, b):
        out[name] = {"kappa": cohen_kappa(a, b), "agree": sum(x == y for x, y in zip(a, b)) / len(a) if a else None, "n": len(a)}

    pair("abstention: verifier vs human",
         [r["verifier_abstained"] for r in labelled], [r["human_abstained"].strip().lower() for r in labelled])

    v_claims, h_claims = [], []
    for r in labelled:
        if r["human_abstained"].strip().lower() == "n" and r["verifier_abstained"] == "n":
            v_claims += _claim_labels(r, "verifier_unsupported")
            h_claims += _claim_labels(r, "human_unsupported")
    pair("claim supported: verifier vs human", v_claims, h_claims)

    pair("overall correct: verifier vs human",
         [r["verifier_correct"] for r in labelled], [r["human_correct"].strip().lower() for r in labelled])
    judged = [r for r in labelled if r["judge_correct"].strip()]
    if judged:
        pair("overall correct: LLM judge vs human",
             [r["judge_correct"].strip().lower() for r in judged], [r["human_correct"].strip().lower() for r in judged])
    return out


def agree(args) -> None:
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    print("| Comparison | n | Agreement | Cohen's κ |\n|---|---|---|---|")
    for name, m in agreement(rows).items():
        acc = "–" if m["agree"] is None else f"{100 * m['agree']:.1f}%"
        kappa = "–" if m["kappa"] is None else f"{m['kappa']:.2f}"
        print(f"| {name} | {m['n']} | {acc} | {kappa} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--config", required=True)
    e.add_argument("--verified", action="append", required=True, help="run=path/to/verified.jsonl")
    e.add_argument("--split", default="test")
    e.add_argument("--n", type=int, default=100, help="Rows per run")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default="results/annotation.csv")
    a = sub.add_parser("agree")
    a.add_argument("--csv", required=True)
    args = parser.parse_args()
    export(args) if args.cmd == "export" else agree(args)


if __name__ == "__main__":
    main()
