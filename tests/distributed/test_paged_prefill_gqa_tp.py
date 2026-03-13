import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from mojo_opset import MojoPagedPrefillGQA, MojoPagedDecodeGQA, MojoStorePagedKVCache
from mojo_opset.distributed.parallel import MojoRowwiseParallel
from torch.distributed.tensor.placement_types import Shard, Replicate
from typing import Optional


class SimplePagedKVCache:
    def __init__(
        self,
        batch_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 2048,
        block_size: int = 16,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cpu")

        max_blocks_per_seq = (max_position_embeddings + self.block_size - 1) // self.block_size
        total_blocks = self.batch_size * max_blocks_per_seq * self.num_layers

        self.k_cache = torch.zeros(
            (total_blocks, self.num_kv_heads, self.block_size, self.head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.v_cache = torch.zeros(
            (total_blocks, self.num_kv_heads, self.block_size, self.head_dim),
            dtype=dtype,
            device=self.device,
        )

        self.block_tables = torch.zeros(
            (self.num_layers, self.batch_size, max_blocks_per_seq),
            dtype=torch.int32,
            device=self.device,
        )

        self.seq_lens = torch.zeros(
            (self.num_layers, self.batch_size), dtype=torch.int64, device=self.device
        )

        self.free_blocks = torch.arange(total_blocks, device=self.device, dtype=torch.int32)
        self.num_free_blocks = total_blocks
        self.store_paged_kv = MojoStorePagedKVCache()

    def _allocate_blocks(self, num_blocks: int):
        if num_blocks > self.num_free_blocks:
            raise ValueError("PagedKVCache: Out of memory!")
        allocated = self.free_blocks[self.num_free_blocks - num_blocks : self.num_free_blocks]
        self.num_free_blocks -= num_blocks
        return allocated

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        input_len: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
    ):
        if input_len is None:
            input_len = torch.ones(self.batch_size, device=key_states.device, dtype=torch.int64)

        current_seq_lens = self.seq_lens[layer_idx]
        for i in range(self.batch_size):
            context_len = current_seq_lens[i].item()

            old_num_blocks = (context_len + self.block_size - 1) // self.block_size
            new_total_len = context_len + input_len[i]
            new_num_blocks = (new_total_len + self.block_size - 1) // self.block_size

            if new_num_blocks > old_num_blocks:
                num_to_allocate = new_num_blocks - old_num_blocks
                newly_allocated = self._allocate_blocks(num_to_allocate)
                self.block_tables[layer_idx, i, old_num_blocks:new_num_blocks] = newly_allocated

        self.store_paged_kv(
            key_states,
            value_states,
            self.k_cache,
            self.v_cache,
            self.block_tables[layer_idx],
            cu_seqlens,
            current_seq_lens,
        )
        self.seq_lens[layer_idx] += input_len

    def get_block_tables_for_decode(self, layer_idx: int):
        max_blocks = (self.seq_lens[layer_idx].max().item() + self.block_size - 1) // self.block_size
        return self.block_tables[layer_idx, :, :max_blocks]


class RefPagedPrefillGQABlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        layer_id: int = 0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale_factor = self.head_dim ** -0.5
        self.attn_prefill = MojoPagedPrefillGQA()
        self.attn_decode = MojoPagedDecodeGQA()
        self.qkv_proj = torch.nn.Linear(
            hidden_size,
            hidden_size + 2 * num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.output_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_input_len: Optional[torch.Tensor] = None,
        context_shifts: Optional[torch.Tensor] = None,
        decode_kv_len: Optional[torch.Tensor] = None,
        context_cu_seqs: Optional[torch.Tensor] = None,
        kv_cache: Optional[SimplePagedKVCache] = None,
    ):
        qkv_projs = self.qkv_proj(hidden_states)
        q_proj, k_proj, v_proj = qkv_projs.split(
            (self.hidden_size, self.num_key_value_heads * self.head_dim, self.num_key_value_heads * self.head_dim),
            dim=-1,
        )

        q = q_proj.contiguous().view(hidden_states.size(0), self.num_attention_heads, self.head_dim)
        k = k_proj.contiguous().view(hidden_states.size(0), self.num_key_value_heads, self.head_dim)
        v = v_proj.contiguous().view(hidden_states.size(0), self.num_key_value_heads, self.head_dim)

        if kv_cache is not None:
            kv_cache.update(
                self.layer_id,
                k,
                v,
                context_input_len,
                context_cu_seqs,
            )

        if context_input_len is not None:
            flash_attn_out = self.attn_prefill(
                q,
                kv_cache.k_cache,
                kv_cache.v_cache,
                context_cu_seqs,
                kv_cache.block_tables[self.layer_id],
                self.scale_factor,
                seqlens_kv=context_shifts + context_input_len,
            )
        else:
            flash_attn_out = self.attn_decode(
                q,
                kv_cache.k_cache,
                kv_cache.v_cache,
                decode_kv_len,
                kv_cache.get_block_tables_for_decode(self.layer_id),
                self.scale_factor,
            )

        flash_attn_out = flash_attn_out.view(hidden_states.size(0), -1)
        flash_attn_out = self.output_proj(flash_attn_out)
        return flash_attn_out


class TPPagedPrefillGQABlock(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        layer_id: int,
        tp_mesh,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale_factor = self.head_dim ** -0.5

        self.qkv_proj = torch.nn.Linear(
            hidden_size,
            hidden_size + 2 * num_key_value_heads * self.head_dim,
            bias=False,
        )

        self.attn_prefill = MojoRowwiseParallel(
            input_layouts=(Replicate(),),
            output_layouts=(Shard(-2),),
            use_local_output=True,
        )(MojoPagedPrefillGQA(), tp_mesh)
        self.attn_decode = MojoRowwiseParallel(
            input_layouts=(Replicate(),),
            output_layouts=(Shard(-2),),
            use_local_output=True,
        )(MojoPagedDecodeGQA(), tp_mesh)

        self.output_proj = MojoRowwiseParallel(
            input_layouts=(Shard(-1),),
            output_layouts=(Replicate(),),
            use_local_output=True,
        )(torch.nn.Linear(hidden_size, hidden_size, bias=False), tp_mesh)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_input_len: Optional[torch.Tensor] = None,
        context_shifts: Optional[torch.Tensor] = None,
        decode_kv_len: Optional[torch.Tensor] = None,
        context_cu_seqs: Optional[torch.Tensor] = None,
        kv_cache: Optional[SimplePagedKVCache] = None,
    ):
        qkv_projs = self.qkv_proj(hidden_states)
        q_proj, k_proj, v_proj = qkv_projs.split(
            (self.hidden_size, self.num_key_value_heads * self.head_dim, self.num_key_value_heads * self.head_dim),
            dim=-1,
        )

        q = q_proj.contiguous().view(hidden_states.size(0), self.num_attention_heads, self.head_dim)
        k = k_proj.contiguous().view(hidden_states.size(0), self.num_key_value_heads, self.head_dim)
        v = v_proj.contiguous().view(hidden_states.size(0), self.num_key_value_heads, self.head_dim)

        if kv_cache is not None:
            kv_cache.update(
                self.layer_id,
                k,
                v,
                context_input_len,
                context_cu_seqs,
            )

        if context_input_len is not None:
            flash_attn_out = self.attn_prefill(
                q,
                kv_cache.k_cache,
                kv_cache.v_cache,
                context_cu_seqs,
                kv_cache.block_tables[self.layer_id],
                self.scale_factor,
                seqlens_kv=context_shifts + context_input_len,
            )
        else:
            flash_attn_out = self.attn_decode(
                q,
                kv_cache.k_cache,
                kv_cache.v_cache,
                decode_kv_len,
                kv_cache.get_block_tables_for_decode(self.layer_id),
                self.scale_factor,
            )

        flash_attn_out = flash_attn_out.view(hidden_states.size(0), -1)
        return self.output_proj(flash_attn_out)


def test_paged_prefill_gqa_tp():
    dist.init_process_group(backend="gloo")
    tp_mesh = init_device_mesh(
        device_type="cpu",
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=["tp"],
    )["tp"]

    tp_rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()

    head_dim = 32
    num_key_value_heads = tp_size
    num_attention_heads = 4 * tp_size
    hidden_size = num_attention_heads * head_dim

    batch_size = 2
    seq_len = 16
    num_layers = 1

    device = torch.device("cpu")
    dtype = torch.float32

    torch.manual_seed(0)
    tp_block = TPPagedPrefillGQABlock(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        layer_id=0,
        tp_mesh=tp_mesh,
    ).to(device=device, dtype=dtype)

    torch.manual_seed(0)
    ref_block = RefPagedPrefillGQABlock(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        layer_id=0,
    ).to(device=device, dtype=dtype)

    kv_cache_tp = SimplePagedKVCache(
        batch_size=batch_size,
        num_layers=num_layers,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=2048,
        block_size=16,
        device=device,
        dtype=dtype,
    )

    kv_cache_ref = None
    if tp_rank == 0:
        kv_cache_ref = SimplePagedKVCache(
            batch_size=batch_size,
            num_layers=num_layers,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=2048,
            block_size=16,
            device=device,
            dtype=dtype,
        )

    torch.manual_seed(1)
    hidden_states = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)
    context_input_len = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)
    context_shifts = torch.zeros_like(context_input_len)
    context_cu_seqs = torch.nn.functional.pad(context_input_len.cumsum(-1), (1, 0))

    out_tp = tp_block(
        hidden_states,
        context_input_len=context_input_len,
        context_shifts=context_shifts,
        context_cu_seqs=context_cu_seqs,
        kv_cache=kv_cache_tp,
    )

    if tp_rank == 0:
        out_ref = ref_block(
            hidden_states,
            context_input_len=context_input_len,
            context_shifts=context_shifts,
            context_cu_seqs=context_cu_seqs,
            kv_cache=kv_cache_ref,
        )
    else:
        out_ref = torch.empty_like(out_tp)
    dist.broadcast(out_ref, src=0)
    torch.testing.assert_close(out_tp, out_ref, rtol=1e-4, atol=1e-4)

    torch.manual_seed(2)
    decode_hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    decode_kv_len = kv_cache_tp.seq_lens[0]

    out_tp_decode = tp_block(
        decode_hidden_states,
        decode_kv_len=decode_kv_len,
        kv_cache=kv_cache_tp,
    )

    if tp_rank == 0:
        out_ref_decode = ref_block(
            decode_hidden_states,
            decode_kv_len=kv_cache_ref.seq_lens[0],
            kv_cache=kv_cache_ref,
        )
    else:
        out_ref_decode = torch.empty_like(out_tp_decode)
    dist.broadcast(out_ref_decode, src=0)
    torch.testing.assert_close(out_tp_decode, out_ref_decode, rtol=1e-4, atol=1e-4)

    dist.destroy_process_group()


if __name__ == "__main__":
    test_paged_prefill_gqa_tp()
