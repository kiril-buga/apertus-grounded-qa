"""Stage 1: LoRA supervised fine-tuning on grounded answers and abstentions.

    python -m groundqa.train_sft --config configs/apertus8b.yaml
"""

import json
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from .config import base_parser, load_config, read_jsonl, seed_everything
from .cpu_fallback import apply_if_no_cuda
from .modeling import last_checkpoint, load_model, load_tokenizer, lora_config, precision_flags, to_prompt_completion


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    s = cfg["sft"]
    apply_if_no_cuda()
    seed_everything(cfg["data"].get("seed", 13))

    data_dir = Path(cfg["data"]["dir"])
    system_in_user = cfg["model"].get("system_in_user", False)
    train = Dataset.from_list([to_prompt_completion(ex, system_in_user) for ex in read_jsonl(data_dir / "train.jsonl")])
    val = Dataset.from_list([to_prompt_completion(ex, system_in_user) for ex in read_jsonl(data_dir / "val.jsonl")])

    tok = load_tokenizer(cfg)
    model = load_model(cfg, for_training=True)

    training_args = SFTConfig(
        output_dir=s["output_dir"],
        num_train_epochs=s["epochs"],
        max_steps=s.get("max_steps", -1),
        learning_rate=float(s["lr"]),
        lr_scheduler_type="cosine",
        warmup_steps=s.get("warmup_ratio", 0.03),  # a float < 1 is a ratio of total steps
        per_device_train_batch_size=s["batch_size"],
        per_device_eval_batch_size=s["batch_size"],
        gradient_accumulation_steps=s["grad_accum"],
        gradient_checkpointing=s.get("gradient_checkpointing", True),
        max_length=s["max_length"],
        completion_only_loss=True,
        **precision_flags(cfg),
        logging_steps=s.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=s.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=s.get("eval_steps", 100),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=s.get("report_to", "none"),
        seed=cfg["data"].get("seed", 13),
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        peft_config=lora_config(cfg),
    )
    trainer.model.print_trainable_parameters()
    resume = last_checkpoint(s["output_dir"])
    if resume:
        print(f"resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(s["output_dir"])
    tok.save_pretrained(s["output_dir"])

    log = Path(s["output_dir"]) / "log_history.json"
    log.write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"saved adapter to {s['output_dir']}")


if __name__ == "__main__":
    main()
