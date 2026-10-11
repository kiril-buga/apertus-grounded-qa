"""Run the whole pipeline on CPU in a few minutes, with no downloads.

It builds a tiny random model with the Apertus architecture (xIELU MLP, QK-norm),
a word-level tokenizer with a chat template and a synthetic QA dataset, then runs scripts/run_all.sh against them
with the lexical scorer in place of the NLI model. The numbers it prints are
meaningless; the point is that every stage runs and hands valid files to the next.

    python scripts/smoke_test.py [--workdir /tmp/groundqa-smoke]
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

NAMES = ["Zorvan", "Kell", "Marsh", "Tibor", "Elna", "Corvi", "Dunmere", "Asko", "Bryn", "Halvard", "Iselt", "Joro"]
THINGS = ["bridge", "library", "observatory", "dam", "tower", "station"]
RIVERS = ["Kell", "Arva", "Sorn", "Melk", "Tavi"]


def synthetic_rows(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        name, thing = f"{rng.choice(NAMES)}{i}", rng.choice(THINGS)
        year, length = rng.randint(1850, 2020), rng.randint(50, 900)
        builder, river = rng.choice(NAMES), rng.choice(RIVERS)
        context = (
            f"The {name} {thing} was built in {year} by engineer {builder}. "
            f"It stands next to the {river} river and is {length} meters long. "
            f"Visitors reach the {name} {thing} by train."
        )
        q, a = rng.choice([
            (f"When was the {name} {thing} built?", f"The {name} {thing} was built in {year}."),
            (f"Who built the {name} {thing}?", f"The {name} {thing} was built by engineer {builder}."),
            (f"How long is the {name} {thing}?", f"The {name} {thing} is {length} meters long."),
        ])
        rows.append({"id": f"syn-{i}", "question": q, "context": context, "answer": a})
    return rows


def build_tokenizer(texts: list[str], out_dir: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=specials))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", eos_token="<|end|>", bos_token=None,
    )
    fast.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
        "{% if add_generation_prompt %}<|assistant|> {% endif %}"
    )
    fast.save_pretrained(out_dir)


def build_model(vocab_size: int, out_dir: Path) -> None:
    import torch
    from transformers import ApertusConfig, ApertusForCausalLM

    torch.manual_seed(0)
    config = ApertusConfig(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
        pad_token_id=0, eos_token_id=5, bos_token_id=None, tie_word_embeddings=True,
    )
    ApertusForCausalLM(config).save_pretrained(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    work = Path(args.workdir or tempfile.mkdtemp(prefix="groundqa-smoke-"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    sys.path.insert(0, str(REPO / "src"))
    from groundqa.prompts import ABSTAIN, SYSTEM_PROMPT

    rows = synthetic_rows(160)
    data_file = work / "synthetic.jsonl"
    data_file.write_text("".join(json.dumps(r) + "\n" for r in rows))

    model_dir = work / "tiny-model"
    texts = [SYSTEM_PROMPT, ABSTAIN, "Context: Question:"] + [f"{r['context']} {r['question']} {r['answer']}" for r in rows]
    build_tokenizer(texts, model_dir)
    from transformers import AutoTokenizer
    build_model(len(AutoTokenizer.from_pretrained(model_dir)), model_dir)

    cfg = yaml.safe_load((REPO / "configs/apertus8b.yaml").read_text())
    cfg["model"].update(name=str(model_dir), dtype="float32", attn_implementation=None)
    cfg["data"].update(source="jsonl", path=str(data_file), dir=str(work / "data"),
                       max_train=96, max_pref=16, max_val=8, max_test=24)
    cfg["lora"].update(r=16, alpha=32)
    cfg["sft"].update(output_dir=str(work / "runs/sft"), epochs=1, max_steps=150, lr=1e-2, batch_size=8, grad_accum=1,
                      max_length=256, eval_steps=50, logging_steps=25, gradient_checkpointing=False, bf16=False)
    cfg["generate"].update(batch_size=8, max_new_tokens=16)
    cfg["verify"].update(backend="lexical", threshold=0.9)
    cfg["prefs"]["samples_per_prompt"] = 2
    cfg["dpo"].update(output_dir=str(work / "runs/dpo"), max_steps=2, batch_size=2, grad_accum=1,
                      max_length=256, eval_steps=1, logging_steps=1, eval_fraction=0.2,
                      gradient_checkpointing=False, bf16=False)
    cfg_path = work / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "TOKENIZERS_PARALLELISM": "false"}
    subprocess.run(["bash", str(REPO / "scripts/run_all.sh"), str(cfg_path), str(work / "runs")],
                   cwd=work, env=env, check=True)

    results = work / "results/results.md"
    assert results.exists(), "report was not written"
    for name in ("base", "sft", "dpo"):
        assert (work / f"runs/verified/{name}.jsonl").stat().st_size > 0, name
    assert (work / "runs/dpo/adapter_config.json").exists(), "DPO adapter missing"
    print(f"\nsmoke test passed; outputs in {work}")


if __name__ == "__main__":
    main()
