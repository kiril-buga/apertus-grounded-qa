"""Stage 2: DPO on verifier-ranked preference pairs.

The SFT adapter is merged into the base weights first. DPO then trains a new
LoRA on top, and the merged SFT model (adapter disabled) is the reference policy.

    python -m groundqa.train_dpo --config configs/apertus8b.yaml \
        --sft-adapter runs/sft --prefs data/rag12k/prefs.jsonl
"""

import json
import random
from pathlib import Path

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from .config import base_parser, load_config, read_jsonl, seed_everything
from .cpu_fallback import apply_if_no_cuda
from .modeling import last_checkpoint, load_model, load_tokenizer, lora_config, precision_flags


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--prefs", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    d = cfg["dpo"]
    seed = cfg["data"].get("seed", 13)
    apply_if_no_cuda()
    seed_everything(seed)

    keep = ("prompt", "chosen", "rejected")
    rows = [{k: r[k] for k in keep} for r in read_jsonl(args.prefs)]
    random.Random(seed).shuffle(rows)
    n_eval = max(1, int(len(rows) * d.get("eval_fraction", 0.05)))
    train, val = Dataset.from_list(rows[n_eval:]), Dataset.from_list(rows[:n_eval])

    tok = load_tokenizer(cfg)
    model = load_model(cfg, adapters=[args.sft_adapter], for_training=True)

    training_args = DPOConfig(
        output_dir=d["output_dir"],
        beta=d.get("beta", 0.1),
        num_train_epochs=d["epochs"],
        max_steps=d.get("max_steps", -1),
        learning_rate=float(d["lr"]),
        lr_scheduler_type="cosine",
        warmup_steps=d.get("warmup_ratio", 0.1),  # a float < 1 is a ratio of total steps
        per_device_train_batch_size=d["batch_size"],
        per_device_eval_batch_size=d["batch_size"],
        gradient_accumulation_steps=d["grad_accum"],
        gradient_checkpointing=d.get("gradient_checkpointing", True),
        max_length=d["max_length"],
        **precision_flags(cfg),
        logging_steps=d.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=d.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=d.get("eval_steps", 50),
        save_total_limit=1,
        report_to=d.get("report_to", "none"),
        seed=seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        peft_config=lora_config(cfg),
    )
    resume = last_checkpoint(d["output_dir"])
    if resume:
        print(f"resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(d["output_dir"])
    (Path(d["output_dir"]) / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"saved DPO adapter to {d['output_dir']} (apply on top of {args.sft_adapter})")


if __name__ == "__main__":
    main()
