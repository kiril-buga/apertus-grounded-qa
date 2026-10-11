import pytest

from groundqa.annotate import agreement
from groundqa.metrics import bootstrap_ci, cohen_kappa, compute_metrics, paired_bootstrap
from groundqa.prefs import build_pairs, reward
from groundqa.prompts import ABSTAIN


def _row(id_, answerable, abstained, n_claims=0, n_supported=0, ref=None, f1=None, sample=0, text="x"):
    return {"id": id_, "sample": sample, "answerable": answerable, "abstained": abstained, "prediction": text,
            "n_claims": n_claims, "n_supported": n_supported, "ref_consistent": ref, "token_f1": f1,
            "claims": [{"text": "c", "p": 1.0, "supported": i < n_supported} for i in range(n_claims)]}


ROWS = [
    _row("1", True, False, 2, 2, True, 0.8),    # correct grounded answer
    _row("2", True, False, 2, 1, True, 0.6),    # one unsupported claim
    _row("3", True, True),                      # over-refusal
    _row("4", False, True),                     # correct abstention
    _row("5", False, False, 1, 0, None, None),  # answered an unanswerable question
]


def test_compute_metrics():
    m = compute_metrics(ROWS)
    assert m["grounded_accuracy"] == pytest.approx(2 / 5)
    assert m["abstain_recall"] == pytest.approx(1 / 2)
    assert m["abstain_precision"] == pytest.approx(1 / 2)
    assert m["over_refusal"] == pytest.approx(1 / 3)
    assert m["claim_support"] == pytest.approx(3 / 5)
    assert m["fully_supported"] == pytest.approx(1 / 3)
    assert m["ref_consistent"] == pytest.approx(1.0)
    assert m["token_f1"] == pytest.approx(0.7)


def test_bootstrap_ci_brackets_point_estimate():
    rows = ROWS * 20
    ci = bootstrap_ci(rows, n_boot=200)
    point = compute_metrics(rows)["grounded_accuracy"]
    assert ci["grounded_accuracy"][0] <= point <= ci["grounded_accuracy"][1]


def test_paired_bootstrap_detects_improvement():
    a = [_row(str(i), True, False, 1, 0, False, 0.2) for i in range(50)]
    b = [_row(str(i), True, False, 1, 1, True, 0.9) for i in range(50)]
    d = paired_bootstrap(a, b, n_boot=200)["grounded_accuracy"]
    assert d["diff"] == pytest.approx(1.0) and d["lo"] > 0


def test_cohen_kappa():
    assert cohen_kappa(["y", "n", "y", "n"], ["y", "n", "y", "n"]) == pytest.approx(1.0)
    assert cohen_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]) == pytest.approx(0.0)


def test_reward_ordering():
    good = reward(_row("a", True, False, 2, 2, True))
    partial = reward(_row("a", True, False, 2, 1, True))
    refusal = reward(_row("a", True, True))
    halluc = reward(_row("a", False, False, 1, 0))
    assert good == 1.0 and good > partial > refusal > halluc == 0.0
    assert reward(_row("a", False, True)) == 1.0


def test_build_pairs_uses_best_and_worst_sample():
    examples = {
        "q1": {"id": "q1", "question": "Q1?", "context": "C1", "answer": "A1", "answerable": True},
        "q2": {"id": "q2", "question": "Q2?", "context": "C2", "answer": None, "answerable": False},
        "q3": {"id": "q3", "question": "Q3?", "context": "C3", "answer": "A3", "answerable": True},
    }
    verified = [
        _row("q1", True, False, 1, 1, True, text="good"),
        _row("q1", True, False, 2, 0, False, text="bad", sample=1),
        _row("q2", False, False, 1, 0, text="made up", sample=0),
        _row("q2", False, False, 1, 0, text="also made up", sample=1),
        _row("q3", True, False, 1, 1, True, text="fine"),
        _row("q3", True, False, 1, 1, True, text="also fine", sample=1),
    ]
    pairs, stats = build_pairs(examples, verified, min_gap=0.5)
    by_id = {p["id"]: p for p in pairs}
    assert by_id["q1"]["chosen"][0]["content"] == "good"
    assert by_id["q1"]["rejected"][0]["content"] == "bad"
    assert by_id["q2"]["chosen"][0]["content"] == ABSTAIN  # gold fallback
    assert "q3" not in by_id  # both samples equally good: no signal
    assert stats["pairs"] == 2 and stats["skipped_small_gap"] == 1


def test_agreement_from_annotation_rows():
    def r(v_abs, h_abs, v_uns, h_uns, v_ok, h_ok, judge=""):
        return {"claims": "[1] a\n[2] b", "verifier_abstained": v_abs, "human_abstained": h_abs,
                "verifier_unsupported": v_uns, "human_unsupported": h_uns,
                "verifier_correct": v_ok, "human_correct": h_ok, "judge_correct": judge}
    rows = [r("n", "n", "2", "2", "n", "n", "y"), r("n", "n", "", "", "y", "y", "y"), r("y", "y", "", "", "y", "y", "n")]
    out = agreement(rows)
    assert out["overall correct: verifier vs human"]["agree"] == 1.0
    assert out["claim supported: verifier vs human"]["n"] == 4
    assert out["overall correct: LLM judge vs human"]["agree"] == pytest.approx(1 / 3)
