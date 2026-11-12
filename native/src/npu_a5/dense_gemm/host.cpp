/*
Derived from DeepGEMM-Ascend (3a35326271b069b8b5bf1a754435acd6cc91575b).
Source: csrc/apis/gemm.hpp
Original SHA256: 09f8c59bd9d2b54bd1a3c5f212bcd27b0621409414b6e1c36bcb81011feacd53
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

#include "host.h"
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include "aot_dispatch.h"
#include "host/scale.h"

namespace mojo::native {
using torch::Tensor;
namespace py = pybind11;

static std::pair<int64_t,int64_t> logical_shape(const Tensor& x) {
    TORCH_CHECK(x.dim() == 2, "dense GEMM operands must be matrices");
    TORCH_CHECK(x.stride(0) == 1 || x.stride(1) == 1,
                "dense GEMM requires one contiguous matrix axis");
    int64_t rows=x.size(0), cols=x.size(1);
    if (x.scalar_type() == at::kChar) {
        TORCH_CHECK(!(x.stride(0)==1 && x.stride(1)==1), "ambiguous FP4 packing axis");
        if (x.stride(1)==1) cols*=2; else rows*=2;
    }
    return {rows,cols};
}

static Tensor dense_gemm(const Tensor& input, const Tensor& weight,
                         const std::optional<Tensor>& input_scale,
                         const std::optional<Tensor>& weight_scale,
                         bool trans_input, bool trans_weight, int output_type,
                         const std::optional<Tensor>& c, const std::optional<float>& alpha,
                         int64_t gran_m, int64_t gran_n, const std::string& compiled_dims) {
    TORCH_CHECK(input.device().type()==c10::DeviceType::PrivateUse1,
                "native dense GEMM requires an NPU tensor");
    c10::OptionalDeviceGuard guard(input.device());
    for (const auto& x : {std::optional<Tensor>(weight),input_scale,weight_scale,c})
        if(x) TORCH_CHECK(x->device()==input.device(), "all GEMM tensors must share a device");
    TORCH_CHECK(output_type==0 || output_type==1, "output must be BF16 or FP32");
    TORCH_CHECK(gran_m>0 && gran_n>0 && gran_m<=INT_MAX && gran_n<=INT_MAX,
                "scale granularities must be positive int32 values");
    TORCH_CHECK(compiled_dims.find_first_not_of("mnk")==std::string::npos,
                "compiled_dims may only contain m, n, k");
    auto a=trans_input ? input.t() : input;
    auto b=trans_weight ? weight : weight.t();
    auto [m,k]=logical_shape(a); auto [n,bk]=logical_shape(b);
    TORCH_CHECK(k==bk, "operand K dimensions differ");
    const bool bf16=a.scalar_type()==at::kBFloat16 && b.scalar_type()==at::kBFloat16;
    const bool lowp=(a.scalar_type()==at::kFloat8_e4m3fn &&
                       (b.scalar_type()==at::kFloat8_e4m3fn || b.scalar_type()==at::kChar)) ||
                      (a.scalar_type()==at::kChar && b.scalar_type()==at::kChar);
    TORCH_CHECK(bf16 || lowp, "supported pairs: BF16/BF16, FP8/FP8, FP8/packed-FP4, packed-FP4/packed-FP4");
    TORCH_CHECK(bf16 ? (!input_scale && !weight_scale) : (input_scale && weight_scale),
                "both scales are required only for low-precision GEMM");
    auto dtype=output_type==0 ? at::kBFloat16 : at::kFloat;
    auto out=torch::empty({m,n}, input.options().dtype(dtype));
    if(c) {
        TORCH_CHECK(c->scalar_type()==dtype && c->dim()==2 && c->size(1)==n &&
                    (c->size(0)==1 || c->size(0)==m), "C must have output dtype and shape [M,N] or [1,N]");
        if(c->size(0)==1) {
            TORCH_CHECK(!alpha, "alpha and row bias cannot be combined");
            TORCH_CHECK(c->is_contiguous(), "row bias must be contiguous");
        } else { out.copy_(*c); }
    }
    if(m==0 || n==0) return out;
    if(k==0) { if(c) out.copy_(*c); else out.zero_(); return out; }
    TORCH_CHECK(m<=UINT32_MAX && n<=UINT32_MAX && k<=UINT32_MAX, "dimensions exceed native uint32 range");
    // Singleton matrices can have both strides equal to one. Make the major
    // axis explicit without changing their logical values or packing.
    for (auto* x : {&a, &b}) {
        if(x->scalar_type()!=at::kChar && x->stride(0)==1 && x->stride(1)==1) {
            auto normalized=torch::empty_strided(x->sizes(), {std::max<int64_t>(x->size(1),2),1}, x->options());
            normalized.copy_(*x); *x=normalized;
        }
    }
    TORCH_CHECK((a.scalar_type()!=at::kChar && b.scalar_type()!=at::kChar) || k%2==0,
                "FP4 GEMM requires an even logical K");
    std::shared_ptr<deep_gemm::EpilogueClass> epilogue = std::make_shared<deep_gemm::IdentityEpilogue>();
    const bool bias = c && c->size(0) == 1;
    if (bias) epilogue = std::make_shared<deep_gemm::BiasEpilogue>(*c);
    else if (alpha) epilogue = std::make_shared<deep_gemm::AlphaEpilogue>(*alpha);
    // Keep transformed tensors alive until their GEMM launch has been queued.
    Tensor sfa, sfb;
    if (!bf16) {
        sfa = deep_gemm::prepare_scale(*input_scale, m, k, gran_m);
        sfb = deep_gemm::prepare_scale(*weight_scale, n, k, gran_n);
    }
    const auto major_a = a.stride(1) == 1 ? deep_gemm::Major::K : deep_gemm::Major::MN;
    const auto major_b = b.stride(1) == 1 ? deep_gemm::Major::K : deep_gemm::Major::MN;
    deep_gemm::GemmDesc desc{
        .a_ptr=a.data_ptr(), .b_ptr=b.data_ptr(), .d_ptr=out.data_ptr(),
        .a_dtype=a.scalar_type(), .b_dtype=b.scalar_type(),
        .sfa_ptr=bf16 ? nullptr : sfa.data_ptr(), .sfb_ptr=bf16 ? nullptr : sfb.data_ptr(),
        .m=static_cast<uint32_t>(m), .n=static_cast<uint32_t>(n), .k=static_cast<uint32_t>(k),
        .num_cores=deep_gemm::get_num_cube_cores(), .cd_dtype=dtype,
        .major_a=major_a, .major_b=major_b, .acc=c.has_value() && !bias,
        .epilogue_class=std::move(epilogue),
        .outer_stride_a=deep_gemm::get_outer_stride(a, major_a),
        .outer_stride_b=deep_gemm::get_outer_stride(b, major_b),
        .outer_stride_d=static_cast<uint64_t>(out.stride(0)),
    };
    const int kind = bf16 ? 0 : a.scalar_type() == at::kChar ? 3 : b.scalar_type() == at::kChar ? 2 : 1;
    deep_gemm::launch_aot_gemm(desc, kind);
    return out;
}

void RegisterDenseGemm(py::module_& m) {
    m.def("dense_gemm", &dense_gemm);
}
} // namespace mojo::native
