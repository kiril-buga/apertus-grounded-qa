from groundqa.prompts import ABSTAIN, is_abstention
from groundqa.verify import LexicalScorer, chunk_context, split_claims, token_f1, verify_rows


def test_abstention_detector():
    assert is_abstention(ABSTAIN)
    assert is_abstention("The context does not mention the launch date.")
    assert is_abstention("I cannot answer that based on the passage.")
    assert is_abstention("Sorry, I can’t answer this.")
    assert not is_abstention("The pump runs at 3000 rpm.")


def test_abstention_detector_first_person_refusals():
    # Seen from the base model: a hedge followed by a refusal.
    assert is_abstention("The concert featured many artists. However, I can't provide the specific lineup "
                         "without more context.")
    assert is_abstention("I'm unable to determine the release date from this text.")
    assert is_abstention("I don't have enough details to answer.")
    assert is_abstention("There is not enough information to say who won.")
    # Factual sentences with similar words are not refusals.
    assert not is_abstention("The backup pump cannot provide full pressure during startup.")
    assert not is_abstention("Users can't find the setting without the admin role.")
    assert not is_abstention("I/O throughput is limited; the controller can't determine the order of writes.")


def test_split_claims_handles_bullets_and_short_fragments():
    text = "The pump runs at 3000 rpm. It uses a 5 kW motor.\n- Cooling is passive.\nYes."
    assert split_claims(text) == [
        "The pump runs at 3000 rpm.",
        "It uses a 5 kW motor.",
        "Cooling is passive. Yes.",
    ]


def test_chunk_context_respects_size_and_overlaps():
    context = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_context(context, max_words=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 25 for c in chunks)
    first_last = chunks[0].split(". ")[-1]
    assert first_last.rstrip(".") in chunks[1]


def test_token_f1():
    assert token_f1("The answer is 42.", "42") == 0.5
    assert token_f1("Nothing", "42") == 0.0


def _ex(id_, context, answer, answerable=True):
    return {"id": id_, "question": "q?", "context": context, "answer": answer, "answerable": answerable}


def test_verify_rows_labels_claims_and_abstentions():
    examples = {
        "a": _ex("a", "The reactor uses molten salt cooling. It was built in 1965.", "It uses molten salt cooling."),
        "b": _ex("b", "Unrelated text about bridges.", None, answerable=False),
        "c": _ex("c", "The capital is Bern.", "Bern"),
    }
    preds = [
        {"id": "a", "prediction": "The reactor uses molten salt cooling. It produces 900 megawatts daily."},
        {"id": "b", "prediction": ABSTAIN},
        {"id": "c", "prediction": "The capital is Bern."},
    ]
    rows = {r["id"]: r for r in verify_rows(examples, preds, LexicalScorer(), threshold=0.99)}

    a = rows["a"]
    assert not a["abstained"] and a["n_claims"] == 2 and a["n_supported"] == 1
    assert [c["supported"] for c in a["claims"]] == [True, False]
    assert a["ref_consistent"] is False  # second claim is not in the reference either

    assert rows["b"]["abstained"] and rows["b"]["n_claims"] == 0 and rows["b"]["ref_consistent"] is None
    assert rows["c"]["ref_consistent"] is True  # short reference: span containment
