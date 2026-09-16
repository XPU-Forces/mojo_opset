"""Small shared checks for public function contracts."""

import torch


def _require_inference(op, *tensors):
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
        raise RuntimeError(f"{op} does not support autograd; use torch.no_grad()")
