from typing import Tuple
import torch


def relative_l2_error(
    output: torch.Tensor, target: torch.Tensor, dim=-1
) -> torch.Tensor:
    return torch.norm(output - target, dim=dim) / torch.norm(target, dim=dim)


def smoothness_loss(
    output: torch.Tensor, bounds: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    start, end = bounds
    dx = torch.linspace(start, end, output.shape[-1], device=output.device)
    return torch.norm(
        torch.gradient(
            torch.gradient(output, spacing=(dx,), dim=-1)[0], spacing=(dx,), dim=-1
        )[0]
    )
