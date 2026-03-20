import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from mojo_opset import (
    MojoPagedPrefillGQA,
    MojoPagedDecodeGQA,
    MojoStorePagedKVCache,
    MojoRoPE,
    MojoRMSNorm,
)
from mojo_opset.distributed.parallel import (
    MojoRowwiseParallel,
    MojoColwiseParallel,
    MojoTensorParallel,
    MojoDataParallel,
    mojo_parallelize_module,
    mojo_parallel_save_state_dict_naive,
    mojo_parallel_load_state_dict_naive,
)
from torch.distributed.tensor.placement_types import Shard, Replicate, Partial
from torch.distributed import breakpoint as dist_breakpoint
from mojo_opset.modeling.config import MojoConfig, BaseModel
from typing import Optional
from tests.dist_common import dist_test


class SimplePagedKVCache(torch.nn.Module):
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
        super().__init__()
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

    def forward(
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

class RotaryEmbedding(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        @lru_cache
        def generate_freqs(max_position_embeddings, inv_freq_fp32):
            t = torch.arange(max_position_embeddings).to(inv_freq_fp32)
            freqs = torch.einsum("i,j->ij", t, inv_freq_fp32)  # T, D/2
            freqs = torch.cat([freqs, freqs], dim=1)  # T,D/2 -> T,D
            return freqs.cos(), freqs.sin()

        head_dim = config.model_config.hidden_size // config.model_config.num_attention_heads

        self.rope_percentage = getattr(config.model_config, "rope_percentage", 1.0)
        self.rope_dim = config.model_config.head_dim * self.rope_percentage

        base = getattr(config.model_config, "rope_base", 10000.0)
        if getattr(config.model_config, "rope_mode", "default") == "ntk":
            base = base * config.model_config.rope_scale ** (
                self.rope_dim / (self.rope_dim - 2)
            )

        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim)),
        )

        cos, sin = generate_freqs(
            config.model_config.max_position_embeddings, self.inv_freq
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.rope = MojoRoPE()

        def load_state_dict_post_hook(module, incompatible_keys) -> None:
            key2ignroe = []
            for miss in incompatible_keys.missing_keys:
                if miss.split('.')[-1] in ("inv_freq", "cos", "sin"):
                    key2ignroe.append(miss)
            # NOTE(liuyuan): incompatible_keys DOES NOT allow to set attribute.
            for key in key2ignroe:
                incompatible_keys.missing_keys.remove(key)
        self.register_load_state_dict_post_hook(load_state_dict_post_hook)

    def _get_pos_embeddings(self, context_input_len, shift):
        bs = shift.size(0)
        kv_len = torch.ones_like(shift) if context_input_len is None else context_input_len
        cos_embeddings = []
        sin_embeddings = []
        for i in range(bs):
            cos_embeddings.append(self.cos[shift[i] : shift[i] + kv_len[i]])
            sin_embeddings.append(self.sin[shift[i] : shift[i] + kv_len[i]])
        return torch.stack(cos_embeddings, dim=0), torch.stack(sin_embeddings, dim=0)

    def forward(self, q, k, cu_seqlens=None, shift=None):
        return self.rope(
            q,
            k,
            self.cos,
            self.sin,
            cu_seqlens=cu_seqlens,
            kv_lens=shift,
            rope_percentage=self.rope_percentage,
            head_first=False,
        )

class FlashAttentionBlock(torch.nn.Module):

    def __init__(self, config, layer_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_id = layer_id
        self.hidden_size = config.model_config.hidden_size
        self.q_heads = config.model_config.num_attention_heads
        self.head_dim = self.hidden_size // self.q_heads
        self.kv_heads = config.model_config.flash_attn_num_kvheads
        self.q_dim = self.head_dim * self.q_heads
        self.kv_dim = self.head_dim * self.kv_heads
        self.scale_factor = self.head_dim ** -0.5

        self.attentions = torch.nn.ModuleDict(
            {"prefill": MojoPagedPrefillGQA(), "decode": MojoPagedDecodeGQA()}
        )

        self.rms_norms = torch.nn.ModuleDict(
            {
                "query": MojoRMSNorm(
                    self.head_dim, eps=config.model_config.rms_norm_eps
                ),
                "key": MojoRMSNorm(self.head_dim, eps=config.model_config.rms_norm_eps),
                "context": (
                    MojoRMSNorm(self.head_dim, eps=config.model_config.rms_norm_eps)
                    if config.model_config.use_context_groupnorm
                    else None
                ),
            }
        )

        self.projs = torch.nn.ModuleDict(
            {
                "qkv": torch.nn.Linear(
                    self.hidden_size,
                    self.q_dim
                    + 2 * self.kv_dim,
                    bias=config.model_config.attention_bias,
                ),
                "output": torch.nn.Linear(
                    self.q_dim,
                    self.hidden_size,
                    bias=config.model_config.attention_bias,
                ),
            }
        )

        self.rope = None
        if self.layer_id + 1 in getattr(config.model_config, "nope_layers", []):
            self.rope = RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_input_len: torch.Tensor = None,
        context_shifts: torch.Tensor = None,
        decode_kv_len: torch.Tensor = None,
        context_cu_seqs: torch.Tensor = None,
        max_seqlen_q: torch.Tensor = None,
        kv_cache: SimplePagedKVCache = None,
    ):
        # dist_breakpoint()
        qkv_projs = self.projs.qkv(hidden_states)
        q_proj, k_proj, v_proj = qkv_projs.split(
            (self.q_dim, self.kv_dim, self.kv_dim), dim=-1
        )
        new_shape = hidden_states.shape[:-1] + (-1, self.head_dim)
        q = q_proj.contiguous().view(new_shape)
        k = k_proj.contiguous().view(new_shape)
        v = v_proj.contiguous().view(new_shape)

        q = self.rms_norms.query(q)
        k = self.rms_norms.key(k)

        if self.rope:
            q, k = self.rope(
                q,
                k,
                context_cu_seqs,
                decode_kv_len - 1 if decode_kv_len is not None else context_shifts,
            )
        
        # dist_breakpoint()

        kv_cache(
            self.layer_id,
            k,
            v,
            context_input_len,
            context_cu_seqs,
        )

        if context_input_len is not None:
            flash_attn_out = self.attentions.prefill(
                q,
                kv_cache.k_cache,
                kv_cache.v_cache,
                context_cu_seqs,
                kv_cache.block_tables[self.layer_id],
                self.scale_factor,
                seqlens_kv=context_shifts + context_input_len,
            )
        else:
            # dist_breakpoint()
            flash_attn_out = self.attentions.decode(
                q,
                kv_cache.k_cache,
                kv_cache.v_cache,
                decode_kv_len,
                kv_cache.get_block_tables_for_decode(self.layer_id),
                self.scale_factor,
            )
        
        if self.rms_norms.context:
            flash_attn_out = self.rms_norms.context(flash_attn_out)
        # dist_breakpoint()

        flash_attn_out = flash_attn_out.view(hidden_states.size(0), -1)
        flash_attn_out = self.projs.output(flash_attn_out)
        return flash_attn_out

class SimpleModelConfig(BaseModel):
    num_attention_heads: int = 32
    hidden_size:int = 4096
    flash_attn_num_kvheads: int = 8
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    node_layers: list = []
    use_context_groupnorm: bool = True

class TestMojoParallel:
    TP_SIZE=2

    @staticmethod
    def make():
        config = MojoConfig(model_config=SimpleModelConfig())
        config.parallel_config.tp_size = TestMojoParallel.TP_SIZE

        device_mesh = init_device_mesh(
            "cpu", (config.parallel_config.tp_size,), mesh_dim_names=["tp"]
        )
        parallel_plan = {
            "projs.qkv": MojoColwiseParallel(use_local_output=False),
            "rms_norms.query": MojoDataParallel(
                desired_args_input_layouts=[Shard(-2)],
                desired_output_layouts=[Shard(-2)],
                use_local_output=False,
            ),
            "rms_norms.key": MojoDataParallel(
                desired_args_input_layouts=[Shard(-2)],
                desired_output_layouts=[Shard(-2)],
                use_local_output=False,
            ),
            "attentions.prefill": MojoTensorParallel(
                input_layouts=[Replicate()],
                output_layouts=[Shard(-2)],
                use_local_output=False,
            ),
            "attentions.decode": MojoTensorParallel(
                input_layouts=[Replicate()],
                output_layouts=[Shard(-2)],
                use_local_output=False,
            ),
            "rms_norms.context": MojoDataParallel(
                desired_args_input_layouts=[Shard(-2)],
                desired_output_layouts=[Shard(-2)],
                use_local_output=False,
            ),
            "projs.output": MojoRowwiseParallel(
                input_layouts=[Shard(-1)],
                output_layouts=[Replicate()],
            ),
        }

        block = FlashAttentionBlock(config, layer_id=0)
        block.apply(TestMojoParallel._init_weight)

        parallel_block = mojo_parallelize_module(
            block,
            device_mesh=device_mesh,
            parallelize_plan=parallel_plan,
        )

        if device_mesh.get_rank() == 0:
            print(parallel_block)
        
        return(config, parallel_block, device_mesh)

    @staticmethod
    def _init_weight(module):
        for p in module.parameters():
            if torch.is_floating_point(p):
                torch.nn.init.ones_(p)


    @dist_test(world_size=TP_SIZE, backend="gloo")
    def test_paged_gqa_tp(self):
        config, parallel_block, device_mesh = self.make()
        # NOTE(liuyuan): We have to make replicate input here because we have already sharded the kv_cache by head  mannually.
        # TODO(liuyuan): Use DTensor for k_cache and v_cache in SimplePagedKVCache as well. Register the Partition func for it.
        kv_cache_tp = MojoDataParallel(
            desired_args_input_layouts=[Replicate(), Replicate(), Replicate()],
            desired_kwargs_input_layouts={
                "input_len": Replicate(),
                "cu_seqlens": Replicate(),
            },
            desired_output_layouts=[],
            module=SimplePagedKVCache(
                batch_size=1,
                num_layers=1,
                num_kv_heads=config.model_config.flash_attn_num_kvheads,
                head_dim=config.model_config.hidden_size
                // config.model_config.num_attention_heads,
            ),
            device_mesh=device_mesh,
        )

        input_tensor_decode = torch.ones(1, config.model_config.hidden_size)
        output_decode = parallel_block(
            input_tensor_decode,
            kv_cache=kv_cache_tp,
            decode_kv_len=torch.ones(1, dtype=torch.int64),
        )

        input_tensor_prefill = torch.ones(128, config.model_config.hidden_size)
        context_input_len = torch.tensor([128], dtype=torch.int64)
        context_shifts = torch.zeros(1, dtype=torch.int64)
        context_cu_seqs = torch.tensor([0, 128], dtype=torch.int64)

        output_prefill = parallel_block(
            input_tensor_prefill,
            kv_cache=kv_cache_tp,
            context_input_len=context_input_len,
            context_shifts=context_shifts,
            context_cu_seqs=context_cu_seqs,
        )

        if device_mesh.get_local_rank("tp") == 0:
            kv_cache_ref = SimplePagedKVCache(
                batch_size=1,
                num_layers=1,
                num_kv_heads=config.model_config.flash_attn_num_kvheads,
                head_dim=config.model_config.hidden_size
                // config.model_config.num_attention_heads,
            )
            ref_block = FlashAttentionBlock(config, layer_id=0)
            ref_block.apply(self._init_weight)

            output_decode_ref = ref_block(
                input_tensor_decode,
                kv_cache=kv_cache_ref,
                decode_kv_len=torch.ones(1, dtype=torch.int64),
            )
            output_prefill_ref = ref_block(
                input_tensor_prefill,
                kv_cache=kv_cache_ref,
                context_input_len=context_input_len,
                context_shifts=context_shifts,
                context_cu_seqs=context_cu_seqs,
            )

            from triton.testing import assert_close
            assert_close(output_decode, output_decode_ref)
            assert_close(output_prefill, output_prefill_ref)
    
    @dist_test(world_size=TP_SIZE, backend="gloo")
    def test_save_and_load(self):
        _, parallel_block, device_mesh = self.make()
        from tempfile import NamedTemporaryFile
        import os
        with NamedTemporaryFile(mode="w+b") as f:
            weight = mojo_parallel_save_state_dict_naive(parallel_block, f)
            print(f"file size: {os.path.getsize(weight)}")
            mojo_parallel_load_state_dict_naive(parallel_block, weight, device_mesh)
            dist.barrier()


if __name__ == "__main__":
    test = TestMojoParallel()
    test.test_save_and_load()
