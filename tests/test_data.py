from groundqa.data import add_unanswerable, prepare_splits, split_by_group


def _examples(n_groups=20, per_group=3):
    out = []
    for g in range(n_groups):
        for k in range(per_group):
            out.append({
                "id": f"{g}-{k}",
                "question": f"What does widget{g} do in case {k}?",
                "context": f"Document {g}. The widget{g} controls valve{g}. Note {k}.",
                "answer": f"It controls valve{g}.",
                "answerable": True,
                "group": f"g{g}",
                "source": "test",
            })
    return out


def test_split_by_group_has_no_document_overlap():
    splits = split_by_group(_examples(), {"train": 0.7, "val": 0.1, "test": 0.2}, seed=1)
    groups = {name: {ex["group"] for ex in rows} for name, rows in splits.items()}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["val"]
    assert not groups["val"] & groups["test"]
    assert sum(len(r) for r in splits.values()) == 60


def test_split_is_deterministic():
    a = split_by_group(_examples(), {"train": 0.8, "test": 0.2}, seed=7)
    b = split_by_group(_examples(), {"train": 0.8, "test": 0.2}, seed=7)
    assert [e["id"] for e in a["test"]] == [e["id"] for e in b["test"]]


def test_add_unanswerable_swaps_context_from_another_document():
    rows = _examples()
    out = add_unanswerable(rows, fraction=0.3, seed=0)
    swapped = [r for r in out if not r["answerable"]]
    assert len(out) == len(rows)
    assert len(swapped) == round(0.3 * len(rows))
    originals = {r["id"]: r for r in rows}
    for r in swapped:
        src = originals[r["swapped_from"]]
        assert r["answer"] is None
        assert r["group"] != src["group"]
        assert r["context"] != src["context"]
        assert r["question"] == src["question"]


def test_hard_negative_prefers_lexical_overlap():
    rows = [
        {"id": "a", "question": "How hot does the turbine blade get?", "context": "Turbine blade coatings.",
         "answer": "x", "answerable": True, "group": "1", "source": "t"},
        {"id": "b", "question": "q", "context": "The turbine blade alloy is nickel based.",
         "answer": "x", "answerable": True, "group": "2", "source": "t"},
        {"id": "c", "question": "q", "context": "Bananas are yellow.",
         "answer": "x", "answerable": True, "group": "3", "source": "t"},
    ]
    # Convert everything; for "a" the donor must be "b", which shares "turbine blade".
    out = add_unanswerable(rows, fraction=1.0, seed=0)
    a = next(r for r in out if r.get("swapped_from") == "a")
    assert a["context"].startswith("The turbine blade alloy")


def test_prepare_splits_keeps_negatives_within_split():
    cfg = {"data": {"splits": {"train": 0.6, "pref": 0.2, "test": 0.2}, "seed": 3, "unanswerable_fraction": 0.3}}
    splits = prepare_splits(cfg, examples=_examples(n_groups=30))
    contexts = {name: {r["context"] for r in rows} for name, rows in splits.items()}
    assert not contexts["train"] & contexts["test"]
    assert not contexts["pref"] & contexts["test"]
    assert any(not r["answerable"] for r in splits["test"])
