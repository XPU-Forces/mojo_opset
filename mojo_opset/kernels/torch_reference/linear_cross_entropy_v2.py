import torch
import torch.nn.functional as F

def linear_cross_entropy_v2_fwd(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
    calc_acc: bool,
    align_precision: bool,
):
    del align_precision
    logits = torch.matmul(inputs.to(weight.dtype).float(), weight.float().t())
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    valid = labels != ignore_index
    lse = torch.logsumexp(logits, dim=-1)
    lse_abs = torch.logsumexp(logits.abs(), dim=-1)
    zloss = (lse_abs.square() * z_loss_weight).masked_fill(~valid, 0.0)
    loss = loss + zloss
    accuracy = torch.empty(0, dtype=torch.float32, device=inputs.device)
    if calc_acc:
        accuracy = (logits.argmax(dim=-1) == labels).float().masked_fill(~valid, 0.0)
    return loss, zloss, accuracy, lse, lse_abs


def linear_cross_entropy_v2_bwd(
    grad_loss: torch.Tensor,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    lse_abs: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
):
    del lse, lse_abs
    inputs_with_grad = inputs.detach().clone().requires_grad_(True)
    weight_with_grad = weight.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        loss, *_ = linear_cross_entropy_v2_fwd(
            inputs_with_grad,
            weight_with_grad,
            labels,
            ignore_index,
            z_loss_weight,
            False,
            True,
        )
    return torch.autograd.grad(loss, (inputs_with_grad, weight_with_grad), grad_loss)


# The two public interfaces share the reference algorithm but remain distinct
# dispatch keys. Their signatures intentionally stay identical.
linear_cross_entropy_v2_and_zloss_fwd = linear_cross_entropy_v2_fwd
linear_cross_entropy_v2_and_zloss_bwd = linear_cross_entropy_v2_bwd
