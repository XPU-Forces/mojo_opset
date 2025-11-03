import os


class SingletonMeta(type):
    """put here temporary, will be removed later"""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def get_forward_mode():
    # e.g. STD:0,1,2, STD stands for mode, 0,1,2 stands for layer index
    mode_str = os.environ.get("MOJO_NORM_FORWARD_MODE", "")
    if mode_str == "":
        return ("STD", [])

    parts = mode_str.split(":", 1)
    mode = parts[0]

    if len(parts) > 1:
        content = parts[1]
        if content:
            layer_idx = [int(num) for num in content.split(",") if num.strip()]
            return (mode, layer_idx)
        else:
            return (mode, [])
    else:
        return (mode, [])


def get_mojo_exec_mode(op_type: str, mode: str, layer_idx: int) -> str:
    """
    Get the execution mode for a specific operator type and layer index.
        e.g. STD:0,1,2, STD stands for mode, 0,1,2 stands for layer index

    Args:
        op_type (str): The type of the operator.
        mode (str): The execution mode, either "FWD" or "BWD".
        layer_idx (int): The index of the layer.

    Returns:
        str: The execution mode for the operator at the given layer index.
    """
    assert mode.upper() in ["FWD", "BWD"]

    mode_str = os.environ.get(f"{op_type.upper()}_{mode.upper()}_MODE", "")
    if mode_str == "":
        return "STD"

    parts = mode_str.split(":", 1)
    mode = parts[0]

    assert mode.upper() in ["STD", "REF", "DUMP", "DIFF", "ANALYZE"]

    if len(parts) > 1:
        content = parts[1]
        if content:
            preset_layer_idx = [int(num) for num in content.split(",") if num.strip()]
            if layer_idx in preset_layer_idx:
                return mode
            else:
                return "STD"
        else:
            return mode
    else:
        return mode
