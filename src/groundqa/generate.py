"""Generate answers for a data split, with the base model or with merged adapters.

    # baseline
    python -m groundqa.generate --config configs/apertus8b.yaml --out runs/preds/base.jsonl
    # after SFT
    python -m groundqa.generate --config configs/apertus8b.yaml --adapter runs/sft --out runs/preds/sft.jsonl
    # k samples per prompt on train, for building preference pairs
    python -m groundqa.generate --config configs/apertus8b.yaml --adapter runs/sft --split train \
        --limit 2000 --num-samples 4 --temperature 0.9 --out runs/preds/sft_train_samples.jsonl
"""

from pathlib import Path

import torch
from tqdm import tqdm

from .config import base_parser, load_config, read_jsonl, seed_everything, write_jsonl
from .modeling import chat_prompt, load_model, load_tokenizer


@torch.inference_mode()
def _generate_batch(model, tok, batch: list[str], max_new_tokens: int, num_samples: int,
                    temperature: float) -> list[list[str]]:
    sample = temperature > 0
    enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=sample,
        temperature=temperature if sample else None,
        top_p=0.95 if sample else None,
        top_k=None,
        num_return_sequences=num_samples,
        pad_token_id=tok.pad_token_id,
    )
    texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [[t.strip() for t in texts[i * num_samples:(i + 1) * num_samples]] for i in range(len(batch))]


def generate(model, tok, prompts: list[str], max_new_tokens: int, num_samples: int = 1,
             temperature: float = 0.0, batch_size: int = 8) -> list[list[str]]:
    """`batch_size` counts sequences, so k samples per prompt means batch_size // k prompts per batch.
    On out-of-memory the batch is halved and retried, down to one prompt."""
    tok.padding_side = "left"
    step = max(1, batch_size // num_samples)
    outputs, start = [], 0
    with tqdm(total=len(prompts), desc="generate") as bar:
        while start < len(prompts):
            batch = prompts[start:start + step]
            try:
                outputs.extend(_generate_batch(model, tok, batch, max_new_tokens, num_samples, temperature))
            except torch.cuda.OutOfMemoryError:
                if step == 1:
                    raise
                torch.cuda.empty_cache()
                step = max(1, step // 2)
                tqdm.write(f"out of GPU memory; retrying with {step} prompt(s) per batch")
                continue
            start += len(batch)
            bar.update(len(batch))
    return outputs


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--adapter", action="append", default=[], help="Adapter dir(s), merged in order")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    g = cfg["generate"]
    seed_everything(cfg["data"].get("seed", 13))

    rows = read_jsonl(Path(cfg["data"]["dir"]) / f"{args.split}.jsonl")[: args.limit]
    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapters=args.adapter).eval()

    prompts = [chat_prompt(tok, ex, cfg) for ex in rows]
    completions = generate(
        model, tok, prompts, g["max_new_tokens"],
        num_samples=args.num_samples, temperature=args.temperature, batch_size=g["batch_size"],
    )

    out = []
    for ex, samples in zip(rows, completions):
        for k, text in enumerate(samples):
            out.append({
                "id": ex["id"],
                "sample": k,
                "prediction": text,
                "adapters": args.adapter,
            })
    write_jsonl(args.out, out)
    print(f"wrote {len(out)} predictions to {args.out}")


if __name__ == "__main__":
    main()
