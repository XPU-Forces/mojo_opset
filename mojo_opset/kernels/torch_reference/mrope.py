import torch


def mrope_fwd(q, k, cos, sin, mrope_section, is_interleaved, head_dim):
    rope_dim = sum(mrope_section) * 2
    head_dim = rope_dim if head_dim is None else head_dim
    half = rope_dim // 2
    if cos.ndim == 3:
        if is_interleaved:
            tables = []
            for table in (cos, sin):
                result = table[0].clone()
                result[..., 1 : mrope_section[1] * 3 : 3] = table[1, ..., 1 : mrope_section[1] * 3 : 3]
                result[..., 2 : mrope_section[2] * 3 : 3] = table[2, ..., 2 : mrope_section[2] * 3 : 3]
                tables.append(result)
            cos, sin = tables
        else:
            cos, sin = (
                torch.cat([section[index] for index, section in enumerate(table.split(mrope_section, -1))], -1)
                for table in (cos, sin)
            )
    cos, sin = cos.view(q.shape[0], 1, half), sin.view(q.shape[0], 1, half)
    for x in (q, k):
        rows = x.view(x.shape[0], -1, head_dim)
        left, right = rows[..., :half].clone(), rows[..., half:rope_dim].clone()
        rows[..., :half] = left * cos - right * sin
        rows[..., half:rope_dim] = right * cos + left * sin
