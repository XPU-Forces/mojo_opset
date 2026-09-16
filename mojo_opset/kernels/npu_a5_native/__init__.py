"""A5 native wrappers backed by optional mojo_opset_lib SKU packages."""

OPS = {}

OVERRIDES = {"npu.a5.950pr": {"native_swa_infer": "sku_950pr.swa"}}
