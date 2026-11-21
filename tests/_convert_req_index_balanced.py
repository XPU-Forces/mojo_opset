"""Synthetic GLM-5 cases; recipe matches the frozen A2/A5/L20 payloads."""

import json

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

CASES = json.loads((HERE / "data/convert_req_index_balanced14.json").read_text())


def small_inputs(c):
    seq = c["seq_lens"]
    query = c["query_lens"]
    r = len(seq)
    m = c["max_blocks_per_req"]
    bs = c["block_size"]
    req = np.repeat(np.arange(r, dtype=np.int32), query)
    positions = np.concatenate([np.arange(s - q, s, dtype=np.int64) for s, q in zip(seq, query)])
    table = np.zeros((r, m), dtype=np.int32)
    physical = 1
    for i, s in enumerate(seq):
        blocks = (s + bs - 1) // bs
        table[i, :blocks] = np.arange(physical, physical + blocks, dtype=np.int32)
        physical += blocks
    assert len(req) == c["num_tokens"] and int(table.max()) * bs + bs < 2**31
    return req, table, positions


def token_chunk(c, req, positions, start, end):
    p = positions[start:end, None]
    length = p + 1
    k = c["num_topk_tokens"]
    j = np.arange(k, dtype=np.int64)[None, :]
    offset = (
        np.int64(c["seed"]) * 1103515245 + p * 12345 + req[start:end, None].astype(np.int64) * 2654435761
    ) % length
    base = np.where(length >= k, (j * length) // k, j)
    result = np.where(j < np.minimum(k, length), (base + offset) % length, -1).astype(np.int32)
    assert np.all((result == -1) | ((result >= 0) & (result <= p)))
    return np.ascontiguousarray(result)


def build_inputs(c, device):
    req, table, positions = small_inputs(c)
    k = c["num_topk_tokens"]
    t = c["num_tokens"]
    tok = torch.empty((t, k), device=device, dtype=torch.int32)
    for start in range(0, t, 1024):
        end = min(t, start + 1024)
        chunk = token_chunk(c, req, positions, start, end)
        tok[start:end].copy_(torch.from_numpy(chunk))
    return [
        torch.from_numpy(req).to(device),
        torch.from_numpy(table).to(device),
        tok,
        c["block_size"],
        k,
        False,
        None,
        None,
    ]
