import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional, Union

from src.horama.plots import plot_maco


# Simple visualization for ranked concepts
def visualize_ranked_concepts(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    pred_label: str,
    pred_prob: float,
    ranked_concepts: List[Tuple[str, float]],
    save_path: str,
    maco_image: Optional[Union[torch.Tensor, np.ndarray]] = None,
    maco_alpha: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: Optional[str] = None,
    topk: int = 10,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Save a side-by-side image with prediction and top-k concept scores."""
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
    elif isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    img_np = np.clip(img_np, 0.0, 1.0)

    lines = ["Selected concepts (similarity)"]
    for rank, (concept, score) in enumerate(ranked_concepts[:topk], start=1):
        lines.append(f"{rank:02d}. {concept} ({score:.4f})")

    with_maco = maco_image is not None and maco_alpha is not None
    ncols = 3 if with_maco else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize if not with_maco else (16, 5))
    if title:
        fig.suptitle(title)

    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title(
        f"Input  |  Pred: {pred_label} ({pred_prob:.4f})",
        fontsize=10,
    )

    text_ax = axes[1]
    if with_maco:
        plot_maco(maco_image, maco_alpha, ax=axes[1])
        axes[1].set_title("MACO", fontsize=10)
        text_ax = axes[2]

    text_ax.axis("off")
    text_ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#E8F2FF",
            edgecolor="#1F5FBF",
            linewidth=1.5,
        ),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
