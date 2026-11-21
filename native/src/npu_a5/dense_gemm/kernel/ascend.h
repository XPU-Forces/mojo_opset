/*
Derived from DeepGEMM-Ascend (3a35326271b069b8b5bf1a754435acd6cc91575b).
Source: include/deep_gemm/ascend.hpp
Original SHA256: 3044ca8c0186d61a18e408c754f411eb88ae6e5c3081eba6b96d3a85c8fe27b1
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

#include <c_api/asc_simd.h>
#include <kernel/common.h>

// Device primitives, split by abstraction layer:
//   tile_ptr — operand_ptr / result_ptr (shape-carrying device pointers) + l*_ptr aliases
//   copy     — GM/L1/L0/UB data movement, scale-factor load, mad, quant-mode inference
//   sync     — intra-block / flag / ffts synchronization
#include <kernel/ascend/tile_ptr.h>
#include <kernel/ascend/copy.h>
#include <kernel/ascend/sync.h>
