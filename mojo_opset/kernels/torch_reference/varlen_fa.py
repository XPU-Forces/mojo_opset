import math

import torch


def varlen_fa_infer_fwd(q, k, v, cu_q_lens, cu_k_lens, is_causal, softmax_scale, gqa_interleave):
    """FP32 reference, including original uneven AABB/ABAB head mapping."""
    cq, ck = cu_q_lens.cpu().tolist(), cu_k_lens.cpu().tolist()
    if cq[0] != 0 or ck[0] != 0 or cq[-1] != q.shape[0] or ck[-1] != k.shape[0]:
        raise ValueError("cu_lens must start at zero and end at the token counts")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax_scale must be positive and finite")
    nq, nk = q.shape[1], k.shape[1]
    if gqa_interleave:
        heads = torch.arange(nq, device=q.device) % nk
    else:
        groups = [nq // nk + (h < nq % nk) for h in range(nk)]
        heads = torch.tensor([h for h, count in enumerate(groups) for _ in range(count)], device=q.device)
    outputs = []
    for qa, qb, ka, kb in zip(cq, cq[1:], ck, ck[1:]):
        if qb <= qa or kb <= ka or (is_causal and qb - qa > kb - ka):
            raise ValueError("sequences must be nonempty, and causal attention requires Q <= KV")
        query = q[qa:qb].transpose(0, 1).float()
        key = k[ka:kb, heads].transpose(0, 1).float()
        value = v[ka:kb, heads].transpose(0, 1).float()
        scores = query @ key.transpose(-1, -2) * softmax_scale
        if is_causal:
            rows = torch.arange(qb - qa, device=q.device) + (kb - ka) - (qb - qa)
            cols = torch.arange(kb - ka, device=q.device)
            scores.masked_fill_(cols[None, :] > rows[:, None], float("-inf"))
        outputs.append((scores.softmax(-1) @ value).transpose(0, 1).to(q.dtype))
    return torch.cat(outputs)
