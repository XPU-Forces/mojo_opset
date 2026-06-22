import os
import torch

from mojo_opset.utils.platform import get_impl_by_platform
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)

# Try to load custom operators
_custom_ops_loaded = False
try:
    # Try to load custom ops from common locations
    custom_ops_path = os.environ.get('MOJO_CUSTOM_OPS_PATH', None)
    
    if custom_ops_path and os.path.exists(custom_ops_path):
        torch.ops.load_library(custom_ops_path)
        logger.info(f"Loaded custom ops from {custom_ops_path}")
        _custom_ops_loaded = True
    else:
        # Try to import custom_ops module
        try:
            import custom_ops
            logger.info("Loaded custom_ops module")
            _custom_ops_loaded = True
        except ImportError:
            logger.warning("custom_ops module not found, will fallback to reference implementations")
    
    # Verify loaded operators
    if _custom_ops_loaded:
        _available_ops = []
        for op_name in ['npu_quant_lightning_indexer', 'indexer_compress_epilog', 
                        'kv_compress_epilog', 'npu_kv_quant_sparse_attn_sharedkv',
                        'npu_kv_quant_sparse_attn_sharedkv_metadata', 'npu_quant_lightning_indexer_metadata']:
            if hasattr(torch.ops.custom, op_name):
                _available_ops.append(op_name)
        
        if _available_ops:
            logger.info(f"AscendC custom ops available: {', '.join(_available_ops)}")
        else:
            logger.warning("No AscendC custom ops available after loading")
            
except Exception as e:
    logger.warning(f"Failed to load custom ops: {e}")


_op_map = get_impl_by_platform()
globals().update(_op_map)
__all__ = list(_op_map.keys())
