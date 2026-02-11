import argparse
import sys
import random
import torch

from mojo_opset.utils.platform import get_platform

if get_platform() == "npu":
    from torch_npu.contrib import transfer_to_npu

# wan
from wan.configs import WAN_CONFIGS
from generate import generate


EXAMPLE_PROMPT = {
    "ti2v-5B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}

SUPPORTED_SIZES = {"ti2v-5B": ("704*1280", "1280*704", "832*480", "480*832")}

SIZE_CONFIGS = {
    "480*832": (480, 832),
    "832*480": (832, 480),
    "704*1280": (704, 1280),
    "1280*704": (1280, 704),
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
    parser.add_argument(
        "--task", type=str, default="ti2v-5B", choices=list(WAN_CONFIGS.keys()), help="The task to run."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num", type=int, default=None, help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        default=True,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=True, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated video to.")
    parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the video from.")
    parser.add_argument("--use_prompt_extend", action="store_true", default=False, help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.",
    )
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.",
    )
    parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the video.")
    parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="The solver used to sample."
    )
    parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers."
    )
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype", action="store_true", default=True, help="Whether to convert model paramerters dtype."
    )

    args = parser.parse_args()
    _validate_args(args)

    return args


if __name__ == "__main__":
    from mojo_opset.utils.patching import apply_mojo_to_wan2_2

    apply_mojo_to_wan2_2()

    args = _parse_args()
    generate(args)
