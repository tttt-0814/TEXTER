import torch
import numpy as np
from typing import Tuple
from torchvision.ops import roi_align


def preprocess_for_maco_image(
    image: torch.Tensor,
    alpha1: torch.Tensor,
    box_size: Tuple[float, float] = (0.20, 0.25),
    noise_level: float = 0.05,
    percentile_image: float = 1.0,
    percentile_alpha: float = 80,
    number_of_crops_per_iteration: int = 6,
    model_input_size: int = 224,
    device: str = "cpu",
) -> torch.Tensor:
    assert box_size[1] >= box_size[0]
    assert len(image.shape) == 3

    image = image.detach().cpu().numpy()
    alpha1 = alpha1.detach().cpu().numpy()

    image = np.clip(
        image,
        np.percentile(image, percentile_image),
        np.percentile(image, 100 - percentile_image),
    )
    image_normalized = (image - image.min()) / (image.max() - image.min())
    alpha1 = np.mean(alpha1, 0, keepdims=True)
    alpha1 = np.clip(alpha1, None, np.percentile(alpha1, percentile_alpha))
    alpha1 = alpha1 / alpha1.max()
    background = np.ones_like(image_normalized)
    image_alpha1 = alpha1 * image_normalized + (1 - alpha1) * background
    image_alpha1 = torch.from_numpy(image_alpha1).to(device)

    # generate random boxes
    x0 = 0.5 + torch.randn((number_of_crops_per_iteration,), device=device) * 0.15
    y0 = 0.5 + torch.randn((number_of_crops_per_iteration,), device=device) * 0.15
    delta_x = (
        torch.rand((number_of_crops_per_iteration,), device=device)
        * (box_size[1] - box_size[0])
        + box_size[1]
    )
    delta_y = delta_x

    min_coords = torch.zeros_like(x0)
    max_coords = torch.ones_like(x0)
    boxes = (
        torch.stack(
            [
                torch.zeros((number_of_crops_per_iteration,), device=device),
                torch.clamp(
                    x0 - delta_x * 0.5, min=min_coords, max=max_coords - delta_x
                ),  # x_min
                torch.clamp(
                    y0 - delta_y * 0.5, min=min_coords, max=max_coords - delta_y
                ),  # y_min
                torch.clamp(x0 + delta_x * 0.5, min=delta_x, max=max_coords),  # x_max
                torch.clamp(y0 + delta_y * 0.5, min=delta_y, max=max_coords),  # y_max
            ],
            dim=1,
        )
        * image_alpha1.shape[1]
    )
    # # Original boxes
    # boxes = (
    #     torch.stack(
    #         [
    #             torch.zeros((number_of_crops_per_iteration,), device=device),
    #             x0 - delta_x * 0.5,
    #             y0 - delta_y * 0.5,
    #             x0 + delta_x * 0.5,
    #             y0 + delta_y * 0.5,
    #         ],
    #         dim=1,
    #     )
    #     * image.shape[1]
    # )

    cropped_and_resized_images = roi_align(
        image_alpha1.unsqueeze(0),
        boxes,
        output_size=(model_input_size * 2, model_input_size * 2),
    ).squeeze(0)

    cropped_and_resized_images = torch.nn.functional.interpolate(
        cropped_and_resized_images,
        size=(model_input_size, model_input_size),
        mode="bicubic",
        align_corners=True,
        antialias=True,
    )

    # # add normal and uniform noise for better robustness
    # cropped_and_resized_images.add_(
    #     torch.randn_like(cropped_and_resized_images) * noise_level
    # )
    # cropped_and_resized_images.add_(
    #     (torch.rand_like(cropped_and_resized_images) - 0.5) * noise_level
    # )

    return cropped_and_resized_images
