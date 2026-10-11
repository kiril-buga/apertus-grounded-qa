"""Turn k sampled answers per training prompt into DPO preference pairs.

The claim verifier acts as the reward: for each prompt, the best-scoring sample
is "chosen" and the worst is "rejected". Pairs whose rewards are too close are
dropped, since they carry little signal and much verifier noise.

    python -m groundqa.prefs --config configs/apertus8b.yaml \
        --samples runs/preds/sft_train_samples.jsonl --out data/rag12k/prefs.jsonl
"""

from collections import Counter, defaultdict
from pathlib import Path

from .config import base_parser, load_config, read_jsonl, write_jsonl
from .modeling import target_text
from .prompts import prompt_messages
from .verify import make_scorer, verify_rows


def reward(row: dict, over_refusal_reward: float = 0.3) -> float:
    """1.0 is ideal. Answering an unanswerable question scores 0; refusing an answerable one scores `over_refusal_reward`."""
    if not row["answerable"]:
        return 1.0 if row["abstained"] else 0.0
    if row["abstained"]:
        return over_refusal_reward
    if row["n_claims"] == 0:
        return 0.0
    support = row["n_supported"] / row["n_claims"]
    return support * (0.5 + 0.5 * bool(row["ref_consistent"]))


def build_pairs(examples: dict[str, dict], verified: list[dict], min_gap: float = 0.5,
                over_refusal_reward: float = 0.3, gold_fallback: bool = True,
                system_in_user: bool = False) -> tuple[list[dict], Counter]:
    by_id = defaultdict(list)
    for row in verified:
        by_id[row["id"]].append((reward(row, over_refusal_reward), row["prediction"]))

    pairs, stats = [], Counter()
    for ex_id, scored in by_id.items():
        ex = examples[ex_id]
        scored.sort(key=lambda t: t[0])
        (r_lo, rejected), (r_hi, chosen) = scored[0], scored[-1]
        source = "sample"
        if r_hi < 1.0 and gold_fallback:
            # No sample was fully right; the reference answer (or the abstention) is the target.
            r_hi, chosen, source = 1.0, target_text(ex), "gold"
        if r_hi - r_lo < min_gap or chosen.strip() == rejected.strip():
            stats["skipped_small_gap"] += 1
            continue
        stats[f"chosen_from_{source}"] += 1
        stats["answerable" if ex["answerable"] else "unanswerable"] += 1
        pairs.append({
            "id": ex_id,
            "prompt": prompt_messages(ex["context"], ex["question"], system_in_user),
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "chosen_reward": round(r_hi, 4),
            "rejected_reward": round(r_lo, 4),
        })
    stats["pairs"] = len(pairs)
    return pairs, stats


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--samples", required=True, help="Output of generate.py with --num-samples > 1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    p, v = cfg["prefs"], cfg["verify"]

    examples = {ex["id"]: ex for ex in read_jsonl(Path(cfg["data"]["dir"]) / f"{args.split}.jsonl")}
    samples = read_jsonl(args.samples)
    verified = verify_rows(examples, samples, make_scorer(cfg), v.get("threshold", 0.5), v.get("chunk_words", 250))
    pairs, stats = build_pairs(
        examples, verified,
        min_gap=p.get("min_gap", 0.5),
        over_refusal_reward=p.get("over_refusal_reward", 0.3),
        gold_fallback=p.get("gold_fallback", True),
        system_in_user=cfg["model"].get("system_in_user", False),
    )
    write_jsonl(args.out, pairs)
    print(dict(stats))
    print(f"wrote {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
