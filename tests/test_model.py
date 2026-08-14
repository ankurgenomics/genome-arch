"""Smoke tests for the genome-arch model: shapes, gradient flow, and the one
correctness property that's cheap to check without training — that reverse
complementing a sequence twice returns the original sequence."""

import torch

from src.model import GenomeArchModel, VOCAB_SIZE


def test_forward_shape():
    model = GenomeArchModel(seq_len=64, embed_dim=16, block_size=8, num_heads=2, topk=4, num_classes=2)
    x = torch.randint(0, VOCAB_SIZE, (4, 64))
    logits = model(x)
    assert logits.shape == (4, 2), f"expected (4, 2), got {tuple(logits.shape)}"


def test_backward_pass_runs():
    model = GenomeArchModel(seq_len=64, embed_dim=16, block_size=8, num_heads=2, topk=4, num_classes=2)
    x = torch.randint(0, VOCAB_SIZE, (4, 64))
    y = torch.randint(0, 2, (4,))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "no gradients were computed"
    assert all(g == g for g in grad_norms), "found NaN in gradients"  # g == g is False for NaN


def test_reverse_complement_is_involution():
    x = torch.randint(0, VOCAB_SIZE, (4, 64))
    rc_twice = GenomeArchModel.reverse_complement(GenomeArchModel.reverse_complement(x))
    assert torch.equal(x, rc_twice), "reverse_complement applied twice must return the original sequence"


def test_block_conv_rejects_wrong_length():
    model = GenomeArchModel(seq_len=64, embed_dim=16, block_size=8, num_heads=2, topk=4, num_classes=2)
    x_wrong_len = torch.randint(0, VOCAB_SIZE, (2, 63))
    try:
        model(x_wrong_len)
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError for a sequence length not divisible by block_size"


if __name__ == "__main__":
    test_forward_shape()
    test_backward_pass_runs()
    test_reverse_complement_is_involution()
    test_block_conv_rejects_wrong_length()
    print("All smoke tests passed.")
