#pragma once
#include <pybind11/pybind11.h>
namespace mojo::native { void RegisterDenseGemm(pybind11::module_& m); }
