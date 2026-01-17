import os
import functools

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


@functools.cache
def _is_taskqueue_enabled():
    if "TRITON_ENABLE_TASKQUEUE" in os.environ:
        return os.getenv("TRITON_ENABLE_TASKQUEUE").lower() in ("true", "1")
    # if not specified, use torch_npu's env
    return os.getenv("TASK_QUEUE_ENABLE", "1") != "0"


@functools.cache
def _get_npu_raw_stream_func():
    import torch_npu

    # According to torch_npu, the content of a torch.npu.Stream is essentilly an rtStream_t
    # TODO: use CANN API instead of torchnpu

    if _is_taskqueue_enabled():
        if hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
            return torch_npu._C._npu_getCurrentRawStreamNoWait
        else:
            from torch.utils.cpp_extension import load_inline

            TORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))
            npu_torch_extras_src = """
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
static PyObject* get_npu_raw_stream_nowait(PyObject* /* not used */, PyObject* device_idx)
{
    HANDLE_TH_ERRORS
    // TORCH_CHECK(
    //     THPUtils_checkLong(device_index), "invalid argument to getCurrentStream", PTA_ERROR(ErrCode::PARAM));
    int64_t device = PyLong_AsLong(device_idx);
    auto aclrt_stream = c10_npu::getCurrentNPUStream(device).stream(false);
    return PyLong_FromVoidPtr(aclrt_stream);
    END_HANDLE_TH_ERRORS
}

static PyMethodDef npu_torch_extras_methods[] = {
    {"get_npu_raw_stream_nowait", get_npu_raw_stream_nowait, METH_O, "Get NPU raw stream nowait"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef npu_torch_extras_module = {
    .m_methods = npu_torch_extras_methods,
};

PyMODINIT_FUNC PyInit_npu_torch_extras(void)
{
    return PyModuleDef_Init(&npu_torch_extras_module);
}
"""

            # Use load_inline as it saves tedious compile configurations
            npu_torch_extras = load_inline(
                name="npu_torch_extras",
                cpp_sources=npu_torch_extras_src,
                extra_include_paths=[
                    os.path.join(TORCH_NPU_INSTALL_PATH, "include"),
                    os.path.join(TORCH_NPU_INSTALL_PATH, "include", "third_party", "acl", "inc"),
                ],
                extra_ldflags=[f"-L{TORCH_NPU_INSTALL_PATH}/lib", "-ltorch_npu"],
            )
            return npu_torch_extras.get_npu_raw_stream_nowait
    else:
        return torch_npu._C._npu_getCurrentRawStream


@functools.cache
def _is_taskqueue_enabled():
    if "TRITON_ENABLE_TASKQUEUE" in os.environ:
        return os.getenv("TRITON_ENABLE_TASKQUEUE").lower() in ("true", "1")
    # if not specified, use torch_npu's env
    return os.getenv("TASK_QUEUE_ENABLE", "1") != "0"


def patched_get_stream_func(device):
    import torch

    if device is None:
        device = torch.npu.current_device()

    get_stream_func = _get_npu_raw_stream_func()
    return get_stream_func(device)


def apply_ttx_npu_patches():
    import triton

    if triton.runtime.driver.active.get_current_target().backend != "npu":
        logger.info(
            f"Skip patching ttx npu, current backend is {triton.runtime.driver.active.get_current_target().backend}."
        )
    else:
        triton.runtime.driver.active.get_current_stream = patched_get_stream_func
