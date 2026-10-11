"""Compare verified runs in one Markdown table, with 95% bootstrap CIs.

The first run is the baseline; later runs also get a paired difference against it.

    python -m groundqa.report --run base=runs/verified/base.jsonl \
        --run sft=runs/verified/sft.jsonl --run sft+dpo=runs/verified/dpo.jsonl \
        --out results/results.md
"""

import argparse
import json
from pathlib import Path

from .config import read_jsonl
from .metrics import bootstrap_ci, compute_metrics, paired_bootstrap

LABELS = {
    "grounded_accuracy": "Grounded accuracy ↑",
    "abstain_recall": "Abstains when unanswerable ↑",
    "over_refusal": "Refuses when answerable ↓",
    "abstain_f1": "Abstention F1 ↑",
    "claim_support": "Claims supported by context ↑",
    "fully_supported": "Answers with every claim supported ↑",
    "ref_consistent": "Consistent with reference ↑",
    "token_f1": "Token F1 vs reference ↑",
    "avg_claims": "Claims per answer",
}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def render(runs: dict[str, list[dict]], n_boot: int = 1000) -> tuple[str, dict]:
    names = list(runs)
    base = names[0]
    points = {n: compute_metrics(rows) for n, rows in runs.items()}
    cis = {n: bootstrap_ci(rows, n_boot) for n, rows in runs.items()}
    diffs = {n: paired_bootstrap(runs[base], runs[n], n_boot) for n in names[1:]}

    header = "| Metric | " + " | ".join(names) + " | " + " | ".join(f"Δ {n} vs {base}" for n in names[1:]) + " |"
    sep = "|---" * (1 + len(names) + len(names[1:])) + "|"
    lines = [header, sep]
    for key, label in LABELS.items():
        cells = []
        for n in names:
            v, ci = points[n].get(key), cis[n].get(key)
            if key == "avg_claims":
                cells.append("–" if v is None else f"{v:.2f}")
            else:
                cells.append(_pct(v) + (f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else ""))
        for n in names[1:]:
            d = diffs[n].get(key)
            if d is None or key == "avg_claims":
                cells.append("–")
            else:
                mark = " *" if d["lo"] > 0 or d["hi"] < 0 else ""
                cells.append(f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]{mark}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    counts = ", ".join(f"{n}: {len(r)}" for n, r in runs.items())
    lines += [
        "",
        f"Values are percentages with 95% bootstrap CIs ({n_boot} resamples). Examples per run: {counts}.",
        "Δ columns are paired bootstrap differences on shared examples; * marks a CI that excludes 0.",
    ]
    return "\n".join(lines), {"metrics": points, "ci": cis, "paired_vs_" + base: diffs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="name=path/to/verified.jsonl")
    parser.add_argument("--out", default="results/results.md")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name] = [r for r in read_jsonl(path) if r.get("sample", 0) == 0]

    table, raw = render(runs, args.n_boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n")
    out.with_suffix(".json").write_text(json.dumps(raw, indent=2))
    print(table)


if __name__ == "__main__":
    main()
