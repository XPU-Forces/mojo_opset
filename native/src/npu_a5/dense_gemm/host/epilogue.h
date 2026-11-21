/*
Derived from DeepGEMM-Ascend (3a35326271b069b8b5bf1a754435acd6cc91575b).
Source: csrc/jit_kernels/epilogue_class.hpp
Original SHA256: c8b348c72544223c12f04bcfa31d62d36132b619e84d4fc2899bdfa06823b001
Adapted for Mojo A5 AOT dispatch.

MIT License

Copyright (c) 2026 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <torch/torch.h>

#include <kernel/common.h>

namespace deep_gemm {

// Carry runtime arguments for the three epilogues in the AOT catalogue.
class EpilogueClass {
public:
    virtual ~EpilogueClass() = default;

    virtual epilogue::Args make_epilogue_operator_args(const uint32_t& n) const { return {}; }
};

class IdentityEpilogue final : public EpilogueClass {};

class AlphaEpilogue final : public EpilogueClass {
    float alpha;

public:
    explicit AlphaEpilogue(const float& alpha) : alpha(alpha) {}

    epilogue::Args make_epilogue_operator_args(const uint32_t& n) const override {
        return {.alpha = alpha};
    }
};

class BiasEpilogue final : public EpilogueClass {
    epilogue::Args args;

public:
    explicit BiasEpilogue(const torch::Tensor& bias) {
        args.bias = reinterpret_cast<uintptr_t>(bias.data_ptr());
    }

    epilogue::Args make_epilogue_operator_args(const uint32_t& n) const override { return args; }
};

} // namespace deep_gemm
