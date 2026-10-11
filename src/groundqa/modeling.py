"""Model, tokenizer and dataset helpers shared by the training and generation scripts."""

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import ABSTAIN, prompt_messages


def load_tokenizer(cfg: dict):
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: dict, adapters: list[str] | None = None, for_training: bool = False):
    """Load the base model and merge any adapters into it, in order.

    Merging keeps inference simple and lets DPO train a fresh LoRA on top of the
    SFT model, with the merged SFT weights as the frozen reference.
    """
    m = cfg["model"]
    kwargs = {"dtype": getattr(torch, m.get("dtype", "bfloat16"))}
    if m.get("attn_implementation"):
        kwargs["attn_implementation"] = m["attn_implementation"]
    if m.get("load_in_4bit"):
        if adapters:
            raise ValueError("merging adapters into a 4-bit model is not supported; set model.load_in_4bit=false")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto" if not for_training else {"": torch.cuda.current_device()}

    model = AutoModelForCausalLM.from_pretrained(m["name"], **kwargs)
    for path in adapters or []:
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    if for_training:
        model.config.use_cache = False
    return model


def lora_config(cfg: dict) -> LoraConfig:
    lc = cfg["lora"]
    return LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc.get("dropout", 0.05),
        target_modules=lc.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


def target_text(ex: dict) -> str:
    return ex["answer"] if ex["answerable"] else ABSTAIN


def to_prompt_completion(ex: dict, system_in_user: bool = False) -> dict:
    """Conversational prompt/completion row; TRL computes the loss on the completion only."""
    return {
        "prompt": prompt_messages(ex["context"], ex["question"], system_in_user),
        "completion": [{"role": "assistant", "content": target_text(ex)}],
    }


def chat_prompt(tok, ex: dict, cfg: dict) -> str:
    m = cfg["model"]
    return tok.apply_chat_template(
        prompt_messages(ex["context"], ex["question"], m.get("system_in_user", False)),
        tokenize=False,
        add_generation_prompt=True,
        **(m.get("chat_template_kwargs") or {}),
    )
