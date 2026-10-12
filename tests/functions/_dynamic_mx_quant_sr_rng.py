# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the repository.

"""Device probes adapted from cannbot-arena's Dynamic MX Quant SR tests.

Imported only by A5-specific tests, so other providers need no CANNBot-DSL.
The probes expose production RNG outputs without allocating huge inputs.
"""

from cannbotdsl import Buffer, dtypes
from cannbotdsl.channel import Channel
from cannbotdsl.lang import host, kernel, vf
from cannbotdsl.ops import reg
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import MemLoc

from mojo_opset.kernels.npu_a5_cannbotdsl import dynamic_mx_quant_sr as op


@kernel
class CounterWordsProbe:
    def __init__(self):
        self.dst = Channel(MemLoc.UB, (8, 64), dtypes.uint32, depth=1)

    def __call__(self, state, out, base, quant_length):
        seed, block_offset = op._read_rng_state(state)
        axis_groups, reciprocal, row_base, col_base = op._rng_group_geometry(base, quant_length)
        key0, key1 = op._split_uint64(seed)
        dst = self.dst.produce()
        with vf(mode="simd"):
            mask = reg.full_mask()
            ids = reg.varange(0, dtypes.uint32)
            c0, c1, c2, c3 = op._nvfp4_counter_contiguous(
                row_base, col_base, ids, axis_groups, reciprocal, block_offset, mask
            )
            reg.vstore(dst, 0, c0, mask)
            reg.vstore(dst, 64, c1, mask)
            reg.vstore(dst, 128, c2, mask)
            reg.vstore(dst, 192, c3, mask)
            w0, w1, w2, w3 = op._philox_counter10(c0, c1, c2, c3, key0, key1, mask)
            reg.vstore(dst, 256, w0, mask)
            reg.vstore(dst, 320, w1, mask)
            reg.vstore(dst, 384, w2, mask)
            reg.vstore(dst, 448, w3, mask)
        mem_copy(out, self.dst.consume())


@host
def run_counter_words(state, out, base: int, quant_length: int):
    probe = CounterWordsProbe()
    probe[1](state, out, base, quant_length)


@kernel
class SixteenBatchWordsProbe:
    def __init__(self):
        self.bf16 = False
        self.random_words = Buffer(MemLoc.UB, (1, 16 * 256 + 64), dtypes.int32)

    def __call__(self, state, out, base, quant_length):
        seed, block_offset = op._read_rng_state(state)
        mem_copy(self.random_words, out)
        op.prepare_random_words_aos(self, base * 16, 16, quant_length, seed, block_offset)
        mem_copy(out, self.random_words)


@host
def run_sixteen_batch_words(state, out, base: int, quant_length: int):
    probe = SixteenBatchWordsProbe()
    probe[1](state, out, base, quant_length)
