# vLLM / vLLM-Ascend位置插值接入

将本目录两份补丁分别应用到任务书指定的参考版本：vLLM `aa31fa0`、vLLM-Ascend `d5923bc`。补丁以两个独立仓库的根目录为工作目录，不是同一个仓库下的两个子目录。

```bash
# 在vLLM参考版本仓库根目录
 git apply --check /path/to/mojo_opset/integrations/vision_pos_embed_interp/vllm-aa31fa0.patch
 git apply /path/to/mojo_opset/integrations/vision_pos_embed_interp/vllm-aa31fa0.patch
# 在vLLM-Ascend参考版本仓库根目录
 git apply --check /path/to/mojo_opset/integrations/vision_pos_embed_interp/vllm-ascend-d5923bc.patch
 git apply /path/to/mojo_opset/integrations/vision_pos_embed_interp/vllm-ascend-d5923bc.patch
```

推理进程须在已验证的CANN9.2/TorchNPU2.10/Triton-Ascend3.6环境中运行，且可导入本版本mojo_opset。例如在本仓库根目录执行`python -m pip install -e . --no-deps`，Python/Triton实现不要求编译native库。

vLLM补丁在NPU分支调用`mojo_opset.integrations.vllm_vision_pos_embed_interp.interpolate_vllm_grid`。它检查HAS_TRITON、设备与dtype、Triton-Ascend/TorchNPU版本及活动CANN版本，满足能力时一次调用多图融合算子；不满足时保留native逐图路径。非NPU仍使用原有vLLM实现。

vLLM-Ascend补丁移除强制native方法覆盖及不再使用的导入，其他attention、并行与PP补丁保持。编译或kernel执行错误会向调用者抛出，不捕获为native回退。

## 接入方法验证

先从官方来源获取参考源文件，不需要下载模型权重：

```bash
python scripts/vision_pos_embed_interp_references.py --output reference
# 在mojo_opset根目录，使用NPU环境
python integrations/vision_pos_embed_interp/validate_patches.py --reference-dir reference --output patch-validation.json
python -m pytest tests/functions/test_vision_pos_embed_integration.py --mojo-implementation triton
```

验证脚本实际应用补丁、核对强制native覆盖已摘除，并抽取原覆盖方法和补丁后的模型方法，在同一真实NPU输入上比较FP32/BF16结果。该验证覆盖插值接入边界，不加载完整vLLM模型。官方未提供随附精度套件，开发测试沿用上游断言与容差，不宣称替代官方验收。

参考源码的完整commit、固定下载地址与SHA-256见同目录reference_sources.json；下载入口会验证文件哈希，已有匹配文件可直接复用。
