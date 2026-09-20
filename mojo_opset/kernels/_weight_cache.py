"""Runtime cache for read-only, prepared inference weights."""

import weakref

import torch


def _cache_context(weight):
    if weight.device.type == "cpu":
        return True, None
    device = torch.get_device_module(weight.device)
    # Replays must execute preparation, rather than capture a cached constant.
    capturing = getattr(device, "is_current_stream_capturing", None)
    if capturing is None:
        return False, None
    with device.device(weight.device):
        if capturing():
            return False, None
        return True, device.current_stream(weight.device)


class PreparedWeightCache:
    """Keep one preparation per live weight, invalidating on normal tensor writes.

    Call inside a custom-op implementation so version checks also run in compiled
    execution. Inference tensors have no version counter and are never cached.
    Reuse is limited to the preparation stream to avoid cross-stream dependencies.
    As with other PyTorch version-based caches, writes through ``.data`` are not
    supported; use ``copy_`` or ``load_state_dict`` to update a weight.
    """

    def __init__(self, prepare):
        self.prepare = prepare
        self._entries = {}

    def __call__(self, weight, transposed=False):
        cacheable, stream = _cache_context(weight)
        if not cacheable:
            return self.prepare(weight, transposed=transposed)
        try:
            version = weight._version
        except RuntimeError:
            return self.prepare(weight, transposed=transposed)

        key = id(weight)
        signature = (
            version, transposed, weight.data_ptr(), weight.shape, weight.stride(), weight.dtype, weight.device, stream
        )
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is weight and entry[1] == signature:
            return entry[2]

        packed = self.prepare(weight, transposed=transposed)
        # An already prepared layout may return the source or a view of it.
        # Caching that alias would keep the source alive through its own entry.
        if packed.data_ptr() == weight.data_ptr():
            self._entries.pop(key, None)
            return packed

        def discard(ref):
            current = self._entries.get(key)
            if current is not None and current[0] is ref:
                self._entries.pop(key, None)

        self._entries[key] = (weakref.ref(weight, discard), signature, packed)
        return packed
