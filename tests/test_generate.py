import torch

from groundqa.generate import generate


class FakeTok:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, batch, **kwargs):
        ids = torch.tensor([[int(p)] * 3 for p in batch])

        class Enc(dict):
            def to(self, device):
                return self
        return Enc(input_ids=ids, attention_mask=torch.ones_like(ids))

    def batch_decode(self, rows, skip_special_tokens=True):
        return [f"answer-{int(r[0])}" for r in rows]


class FakeModel:
    """Fails with out-of-memory whenever a batch holds more than `max_seqs` sequences."""
    device = "cpu"

    def __init__(self, max_seqs):
        self.max_seqs = max_seqs
        self.batch_sizes = []

    def generate(self, input_ids, num_return_sequences=1, **kwargs):
        n = input_ids.shape[0] * num_return_sequences
        self.batch_sizes.append(n)
        if n > self.max_seqs:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        out = input_ids.repeat_interleave(num_return_sequences, dim=0)
        return torch.cat([out, out[:, :1]], dim=1)  # one "generated" token: the prompt's id


def test_batch_size_counts_sequences_not_prompts():
    model = FakeModel(max_seqs=100)
    out = generate(model, FakeTok(), [str(i) for i in range(10)], 8, num_samples=4, batch_size=16)
    assert max(model.batch_sizes) == 16  # 4 prompts x 4 samples, not 16 x 4
    assert out == [[f"answer-{i}"] * 4 for i in range(10)]


def test_halves_batch_on_out_of_memory_and_keeps_order():
    model = FakeModel(max_seqs=4)
    out = generate(model, FakeTok(), [str(i) for i in range(7)], 8, num_samples=2, batch_size=16)
    assert out == [[f"answer-{i}"] * 2 for i in range(7)]
    assert max(n for n in model.batch_sizes if n <= 4) <= 4
