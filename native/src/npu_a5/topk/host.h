#pragma once

#include <ATen/ATen.h>
#include <optional>
#include <tuple>

namespace mojo::native::a5 {
std::tuple<std::optional<at::Tensor>, at::Tensor> TopKForward(
    const at::Tensor& input, int64_t k, const std::optional<at::Tensor>& end,
    bool sorted_index, const std::optional<at::Tensor>& output_idx,
    const std::optional<at::Tensor>& output_idx_offset, int64_t idx_fill,
    double value_fill, bool return_value, bool abort_when_nan_found);
}
