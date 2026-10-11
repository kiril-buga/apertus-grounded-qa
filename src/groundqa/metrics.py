"""Aggregate metrics over verified rows, with bootstrap confidence intervals."""

import random
from collections import Counter


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def is_grounded_correct(r: dict) -> bool:
    """Headline per-example score: abstain when you should, otherwise answer fully supported and consistent with the reference."""
    if not r["answerable"]:
        return r["abstained"]
    return (not r["abstained"]) and r["n_claims"] > 0 and r["n_supported"] == r["n_claims"] and bool(r["ref_consistent"])


def compute_metrics(rows: list[dict]) -> dict[str, float | None]:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["abstained"]]
    ans_answered = [r for r in ans if not r["abstained"]]
    abstained = [r for r in rows if r["abstained"]]

    recall = _rate(sum(r["abstained"] for r in unans), len(unans))
    precision = _rate(sum(not r["answerable"] for r in abstained), len(abstained))
    if recall is None:
        f1 = None
    elif precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    n_claims = sum(r["n_claims"] for r in answered)
    with_claims = [r for r in answered if r["n_claims"]]
    return {
        "grounded_accuracy": _rate(sum(is_grounded_correct(r) for r in rows), len(rows)),
        "abstain_recall": recall,
        "abstain_precision": precision,
        "abstain_f1": f1,
        "over_refusal": _rate(sum(r["abstained"] for r in ans), len(ans)),
        "claim_support": _rate(sum(r["n_supported"] for r in answered), n_claims),
        "fully_supported": _rate(sum(r["n_supported"] == r["n_claims"] for r in with_claims), len(with_claims)),
        "ref_consistent": _rate(sum(bool(r["ref_consistent"]) for r in ans_answered), len(ans_answered)),
        "token_f1": _rate(sum(r["token_f1"] for r in ans_answered), len(ans_answered)),
        "avg_claims": _rate(n_claims, len(answered)),
    }


def bootstrap_ci(rows: list[dict], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict[str, tuple[float, float] | None]:
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        resample = [rows[rng.randrange(len(rows))] for _ in rows]
        for k, v in compute_metrics(resample).items():
            if v is not None:
                samples.setdefault(k, []).append(v)
    return {k: _percentiles(v, alpha) for k, v in samples.items()}


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], n_boot: int = 1000, alpha: float = 0.05,
                     seed: int = 0) -> dict[str, dict]:
    """Difference B - A on the same examples, resampling example ids jointly."""
    by_id_a = {r["id"]: r for r in rows_a}
    by_id_b = {r["id"]: r for r in rows_b}
    ids = sorted(by_id_a.keys() & by_id_b.keys())
    if not ids:
        raise ValueError("runs share no example ids")
    a = [by_id_a[i] for i in ids]
    b = [by_id_b[i] for i in ids]

    point_a, point_b = compute_metrics(a), compute_metrics(b)
    rng = random.Random(seed)
    diffs: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = [rng.randrange(len(ids)) for _ in ids]
        ma = compute_metrics([a[i] for i in idx])
        mb = compute_metrics([b[i] for i in idx])
        for k in ma:
            if ma[k] is not None and mb[k] is not None:
                diffs.setdefault(k, []).append(mb[k] - ma[k])

    out = {}
    for k, d in diffs.items():
        if point_a[k] is None or point_b[k] is None:
            continue
        lo, hi = _percentiles(d, alpha)
        out[k] = {"diff": point_b[k] - point_a[k], "lo": lo, "hi": hi, "n": len(ids)}
    return out


def cohen_kappa(a: list, b: list) -> float | None:
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def _percentiles(values: list[float], alpha: float) -> tuple[float, float]:
    s = sorted(values)
    lo = s[int((alpha / 2) * (len(s) - 1))]
    hi = s[int((1 - alpha / 2) * (len(s) - 1))]
    return lo, hi
