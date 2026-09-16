import torch

from mojo_opset import modules
from mojo_opset import functions


def main() -> None:
    x = torch.randn(4, 128, requires_grad=True)

    # Public functions and modules share the same dispatch contract.
    y = functions.silu(x, implementation="torch_reference")
    norm = modules.RMSNorm(128, implementation="torch_reference")
    output = norm(y)
    output.sum().backward()

    print(output.shape, x.grad.shape)


if __name__ == "__main__":
    main()
