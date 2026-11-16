/*
Derived from DeepGEMM-Ascend (3a35326271b069b8b5bf1a754435acd6cc91575b).
Source: csrc/utils/math.hpp
Original SHA256: 18afc0d426cb44d3a093954c9a899620813ae78cd1ebe6b66bae59d9d1e3f30b
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

namespace deep_gemm {

constexpr auto ceil_div(auto a, auto b) {
    return (a + b - 1) / b;
}

constexpr auto align(auto a, auto b) {
    return ceil_div(a, b) * b;
}

} // namespace deep_gemm
