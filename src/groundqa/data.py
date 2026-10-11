"""Load grounded QA data, split it without leakage, and add unanswerable cases.

Every example is a dict with:
    id, question, context, answer (str | None), answerable (bool), group, source

`group` identifies the source document. Splits are made by group, so no
context seen in training appears in val or test.
"""

import hashlib
import random
import re
from collections import Counter

from .config import read_jsonl

_STOP = set(
    "the a an and or of to in on for with by from at as is are was were be been "
    "what which who whom whose when where why how does do did this that these "
    "those it its into about than then there their they he she his her you your".split()
)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


# --------------------------------------------------------------------------- loaders

def load_rag12k() -> list[dict]:
    """neural-bridge/rag-dataset-12000: long-form answers, all answerable."""
    from datasets import load_dataset

    out = []
    ds = load_dataset("neural-bridge/rag-dataset-12000")
    for split in ds:
        for i, row in enumerate(ds[split]):
            if not row["context"] or not row["question"] or not row["answer"]:
                continue
            out.append({
                "id": f"rag12k-{split}-{i}",
                "question": row["question"].strip(),
                "context": row["context"].strip(),
                "answer": row["answer"].strip(),
                "answerable": True,
                "group": _hash(row["context"]),
                "source": "rag12k",
            })
    return out


def load_squad_v2() -> list[dict]:
    """rajpurkar/squad_v2: short span answers with native unanswerable questions."""
    from datasets import load_dataset

    out = []
    ds = load_dataset("rajpurkar/squad_v2")
    for split in ds:
        for row in ds[split]:
            texts = row["answers"]["text"]
            out.append({
                "id": f"squad-{row['id']}",
                "question": row["question"].strip(),
                "context": row["context"].strip(),
                "answer": texts[0].strip() if texts else None,
                "answerable": bool(texts),
                "group": _hash(row["title"]),  # whole article, not just the paragraph
                "source": "squad_v2",
            })
    return out


def load_jsonl_source(path: str) -> list[dict]:
    """Your own testbed. Fields: question, context, answer; optional id, answerable, group."""
    out = []
    for i, row in enumerate(read_jsonl(path)):
        answer = row.get("answer") or None
        out.append({
            "id": str(row.get("id", f"jsonl-{i}")),
            "question": row["question"].strip(),
            "context": row["context"].strip(),
            "answer": answer.strip() if answer else None,
            "answerable": bool(row.get("answerable", answer is not None)),
            "group": str(row.get("group") or _hash(row["context"])),
            "source": "jsonl",
        })
    return out


def load_source(source: str, path: str | None = None) -> list[dict]:
    if source == "rag12k":
        return load_rag12k()
    if source == "squad_v2":
        return load_squad_v2()
    if source == "jsonl":
        if not path:
            raise ValueError("data.path is required for source=jsonl")
        return load_jsonl_source(path)
    raise ValueError(f"unknown data source: {source}")


# --------------------------------------------------------------------------- splitting

def split_by_group(examples: list[dict], fractions: dict[str, float], seed: int) -> dict[str, list[dict]]:
    """Assign whole groups to splits so a document never crosses a split boundary."""
    groups = sorted({ex["group"] for ex in examples})
    random.Random(seed).shuffle(groups)
    total = sum(fractions.values())
    names = list(fractions)
    bounds, acc = [], 0.0
    for name in names:
        acc += fractions[name] / total
        bounds.append(round(acc * len(groups)))
    bounds[-1] = len(groups)

    assignment, start = {}, 0
    for name, end in zip(names, bounds):
        for g in groups[start:end]:
            assignment[g] = name
        start = end

    splits = {name: [] for name in names}
    for ex in examples:
        splits[assignment[ex["group"]]].append(ex)
    return splits


def add_unanswerable(examples: list[dict], fraction: float, seed: int, pool_size: int = 50) -> list[dict]:
    """Turn `fraction` of the examples into unanswerable ones by swapping in another
    document's context.

    A random context would be too easy to reject, so the swap picks the candidate
    (out of `pool_size` random ones from other groups) that shares the most content
    words with the question. Call this per split so contexts never leak across splits.
    """
    if fraction <= 0 or len(examples) < 2:
        return list(examples)
    rng = random.Random(seed)
    answerable_idx = [i for i, ex in enumerate(examples) if ex["answerable"]]
    n_convert = min(len(answerable_idx), round(fraction * len(examples)))
    to_convert = set(rng.sample(answerable_idx, n_convert))

    out = []
    for i, ex in enumerate(examples):
        if i not in to_convert:
            out.append(ex)
            continue
        q_words = _content_words(ex["question"])
        candidates = [examples[j] for j in rng.sample(range(len(examples)), min(pool_size, len(examples)))]
        candidates = [c for c in candidates if c["group"] != ex["group"]]
        if not candidates:
            out.append(ex)
            continue
        donor = max(candidates, key=lambda c: len(q_words & _content_words(c["context"])))
        out.append({
            **ex,
            "id": f"{ex['id']}__swap",
            "context": donor["context"],
            "answer": None,
            "answerable": False,
            "group": donor["group"],
            "swapped_from": ex["id"],
        })
    return out


def prepare_splits(cfg: dict, examples: list[dict] | None = None) -> dict[str, list[dict]]:
    d = cfg["data"]
    seed = d.get("seed", 13)
    if examples is None:
        examples = load_source(d["source"], d.get("path"))

    max_chars = d.get("max_context_chars")
    if max_chars:
        examples = [ex for ex in examples if len(ex["context"]) <= max_chars]

    splits = split_by_group(examples, d["splits"], seed)
    rng = random.Random(seed)
    for name, rows in splits.items():
        rng.shuffle(rows)
        cap = d.get(f"max_{name}")
        if cap:
            del rows[cap:]
        splits[name] = add_unanswerable(rows, d.get("unanswerable_fraction", 0.0), seed + len(name))
    return splits


def summarize(splits: dict[str, list[dict]]) -> str:
    lines = []
    for name, rows in splits.items():
        c = Counter(ex["answerable"] for ex in rows)
        groups = len({ex["group"] for ex in rows})
        lines.append(f"{name:>5}: {len(rows):6d} examples | {c[True]} answerable, {c[False]} unanswerable | {groups} documents")
    return "\n".join(lines)
