"""Claim-level verification of answers against their context.

Each answer is split into sentence-level claims. An NLI model checks each claim
against the context (the context is chunked and the best chunk counts). A claim
with entailment probability >= threshold is "supported". Abstentions are
detected separately and have no claims.

    python -m groundqa.verify --config configs/apertus8b.yaml \
        --preds runs/preds/sft.jsonl --out runs/verified/sft.jsonl
"""

import re
import string
from collections import Counter
from pathlib import Path

from .config import base_parser, load_config, read_jsonl, write_jsonl
from .prompts import is_abstention

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        if line:
            out.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
    return out


def split_claims(answer: str, min_words: int = 3) -> list[str]:
    """Sentence-level claims. Fragments shorter than `min_words` are merged into the previous claim."""
    claims: list[str] = []
    for sent in split_sentences(answer):
        if len(sent.split()) < min_words and claims:
            claims[-1] = f"{claims[-1]} {sent}"
        else:
            claims.append(sent)
    return claims


def chunk_context(context: str, max_words: int = 250) -> list[str]:
    """Pack sentences into chunks of at most ~max_words, overlapping by one sentence."""
    sents = split_sentences(context) or [context]
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
            chunks.append(" ".join(cur))
            cur = [cur[-1]]
        cur.append(s)
    chunks.append(" ".join(cur))
    return chunks


# --------------------------------------------------------------------------- scorers

class NLIScorer:
    """Entailment probability from an MNLI-style cross-encoder."""

    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for name, i in labels.items() if name.startswith("entail"))
        self.batch_size = batch_size
        self.max_length = max_length

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                premises, hyps = zip(*pairs[i:i + self.batch_size])
                enc = self.tok(list(premises), list(hyps), truncation="only_first", max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.device)
                probs = self.model(**enc).logits.float().softmax(-1)[:, self.entail_idx]
                out.extend(probs.tolist())
        return out


class LexicalScorer:
    """Share of the hypothesis's content words found in the premise.

    Only for tests and the offline smoke run. It is not a faithfulness metric.
    """

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        out = []
        for premise, hyp in pairs:
            p, h = set(_tokens(premise)), set(_tokens(hyp))
            out.append(len(p & h) / len(h) if h else 1.0)
        return out


def make_scorer(cfg: dict):
    v = cfg["verify"]
    if v.get("backend", "nli") == "lexical":
        return LexicalScorer()
    return NLIScorer(v["nli_model"], batch_size=v.get("batch_size", 32))


# --------------------------------------------------------------------------- reference metrics

def _tokens(text: str) -> list[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    p, r = _tokens(prediction), _tokens(reference)
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- verification

def verify_rows(examples: dict[str, dict], preds: list[dict], scorer, threshold: float,
                chunk_words: int = 250, short_ref_words: int = 5) -> list[dict]:
    """Return one verified row per prediction. Scorer calls are batched across all rows."""
    rows, pairs, owners = [], [], []
    for pred in preds:
        ex = examples[pred["id"]]
        text = pred["prediction"]
        abstained = is_abstention(text)
        claims = [] if abstained else split_claims(text)
        row = {
            "id": pred["id"],
            "sample": pred.get("sample", 0),
            "answerable": ex["answerable"],
            "abstained": abstained,
            "prediction": text,
            "claims": [{"text": c, "p": 0.0} for c in claims],
            "token_f1": None,
            "ref_consistent": None,
        }
        chunks = chunk_context(ex["context"], chunk_words)
        for ci, claim in enumerate(claims):
            for chunk in chunks:
                pairs.append((chunk, claim))
                owners.append((len(rows), "claim", ci))

        if ex["answerable"] and not abstained:
            ref = ex["answer"]
            row["token_f1"] = token_f1(text, ref)
            if len(ref.split()) <= short_ref_words:
                # Short span answers (SQuAD-style): check the span appears in the answer.
                row["ref_consistent"] = " ".join(_tokens(ref)) in " ".join(_tokens(text))
            else:
                # Long answers: every claim must follow from the reference answer.
                for ci, claim in enumerate(claims):
                    pairs.append((ref, claim))
                    owners.append((len(rows), "ref", ci))
                row["_ref_p"] = [0.0] * len(claims)
        rows.append(row)

    for (ri, kind, ci), p in zip(owners, scorer.score(pairs)):
        if kind == "claim":
            rows[ri]["claims"][ci]["p"] = max(rows[ri]["claims"][ci]["p"], p)
        else:
            rows[ri]["_ref_p"][ci] = p

    for row in rows:
        for c in row["claims"]:
            c["p"] = round(c["p"], 4)
            c["supported"] = c["p"] >= threshold
        row["n_claims"] = len(row["claims"])
        row["n_supported"] = sum(c["supported"] for c in row["claims"])
        ref_p = row.pop("_ref_p", None)
        if ref_p is not None:
            row["ref_consistent"] = bool(ref_p) and all(p >= threshold for p in ref_p)
    return rows


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    v = cfg["verify"]

    examples = {ex["id"]: ex for ex in read_jsonl(Path(cfg["data"]["dir"]) / f"{args.split}.jsonl")}
    preds = read_jsonl(args.preds)
    rows = verify_rows(examples, preds, make_scorer(cfg), v.get("threshold", 0.5), v.get("chunk_words", 250))
    write_jsonl(args.out, rows)
    print(f"verified {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
