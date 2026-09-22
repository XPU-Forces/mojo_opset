"""A5 native wrappers backed by mojo_opset_lib SKU extensions."""

OPS = {}

OVERRIDES = {"npu.a5.950pr": {"native_swa_infer": "sku_950pr.swa"}}
