"""Let TRL train on CPU, for the smoke test only.

Recent TRL computes per-token log-probs with a Triton kernel that needs a GPU.
On a machine without CUDA this swaps in a plain PyTorch version with the same
outputs. It materialises the full logits, so it is only fit for tiny models.
"""

import types

import torch


def _torch_logprobs(hidden, weight, bias, targets, temperature, chunk_size,
                    final_logit_softcapping=None, logit_scale=1.0, outputs=("log_probs",)):
    logits = (hidden @ weight.t()).float()
    if bias is not None:
        logits = logits + bias.float()
    logits = logits * logit_scale
    if final_logit_softcapping:
        logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
    logits = logits / temperature
    logp = logits.log_softmax(-1)
    log_probs = logp.gather(-1, targets[:, None]).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1) if "entropy" in outputs else None
    log_sum_sq = torch.logsumexp(2 * logp, -1).detach() if "log_sum_sq_probs" in outputs else None
    mean_logits = logits.mean(-1).detach() if "mean_logits" in outputs else None
    is_top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
    return log_probs, entropy, log_sum_sq, mean_logits, is_top1


def apply_if_no_cuda() -> None:
    if torch.cuda.is_available():
        return
    import trl.trainer.utils as trl_utils

    trl_utils._ChunkedLogProbFunction = types.SimpleNamespace(apply=_torch_logprobs)
