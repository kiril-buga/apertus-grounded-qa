# Runbook: one weekend, one GPU

## Before you start

- A single A100 or H100 (40–80 GB) for the 8B run, plus a 24 GB GPU or the same machine for the 1.5B dev run.
- `huggingface-cli login`, and accept the model terms on the Apertus model pages if prompted.
- `pip install -e ".[dev]" && pytest && python scripts/smoke_test.py` should pass before you spend GPU time.

## Saturday

1. **Dev run on 1.5B** (`bash scripts/run_all.sh configs/apertus1p5b_dev.yaml runs/dev`). The point is to catch
   problems cheaply, not to get results. Check:
   - The chat template accepts a system message. If it errors or renders it oddly, set `model.system_in_user: true`.
   - Read 20 baseline answers next to `is_abstention()`. Does the detector catch every refusal the base model
     phrases its own way? Add patterns to `prompts.py` if not.
   - Read 20 verified rows. Do the claim splits and NLI scores look sane?
2. **8B SFT** (`run_all.sh configs/apertus8b.yaml`, steps 1–4). Watch that eval loss falls and doesn't turn up by
   epoch 2.
3. **Preference data and DPO** (steps 5–6). Check the printed pair stats: if most pairs use `chosen_from_gold`, the
   SFT model rarely gets it fully right, and DPO is closer to SFT on references. In the DPO logs, `rewards/margins`
   should rise and `logps/chosen` should not collapse.

## Sunday

4. **Evaluate** (steps 7–8). Read `results/results.md`. Watch over-refusal: DPO toward abstention can teach the model
   to refuse too often, and `prefs.over_refusal_reward` is the knob for that.
5. **Label** about 60 rows per run with `groundqa.annotate export --n 60`. That takes roughly 2–3 hours, and you
   can't see which model wrote each row. Run `agree`. If you tune `verify.threshold`, tune it on half the labels and
   report agreement on the other half.
6. **Optional:** LLM judge on the same rows (`groundqa.judge`) for verifier vs judge vs human agreement.
7. **Write up.** Fill in the README Results tables, add 3–5 sentences on what changed and what didn't, and include
   one or two concrete examples (a hallucinated claim the base model made that SFT or DPO fixed, and a failure
   that remains). Commit `results/results.md`, `results/results.json` and the agreement table. Commit the adapters
   only if you upload them to the Hugging Face Hub instead of git.

## If something breaks

| Symptom | Likely cause |
|---|---|
| CUDA OOM in SFT | Lower `sft.max_length` or `data.max_context_chars`, or set `model.load_in_4bit: true` |
| CUDA OOM in DPO | `dpo.batch_size: 1` (default) and higher `grad_accum`; lower `dpo.max_length` |
| Base model never "abstains" | The detector misses its phrasing; see Saturday step 1 |
| Very few DPO pairs | Raise `prefs.samples_per_prompt` or temperature, or lower `prefs.min_gap` |
| DPO raises over-refusal | Lower `prefs.over_refusal_reward` so refusals rank lower, or raise `dpo.beta` to stay closer to SFT |
