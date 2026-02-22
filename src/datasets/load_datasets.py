from typing import Tuple, List, Optional

import torchvision
import torchvision.transforms as transforms


def load_dataset(
    dataset: str, data_root: str
) -> Tuple[torchvision.datasets.VisionDataset, transforms.Compose, Optional[List[str]]]:
    """Load dataset and return dataset, transform, and class names."""

    if dataset == "imagenet":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        test_dataset = torchvision.datasets.ImageNet(
            root=data_root, split="val", transform=transform
        )
        try:
            from src.utils.class_labels import IMAGENET_CLASSES

            class_names = IMAGENET_CLASSES
        except ImportError:
            class_names = (
                list(test_dataset.classes)
                if hasattr(test_dataset, "classes")
                else None
            )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Only 'imagenet' is supported."
        )

    return test_dataset, transform, class_names


def select_samples_by_class(
    dataset: torchvision.datasets.VisionDataset,
    target_classes: int,
    images_per_class: int,
) -> List[Tuple[int, str, int]]:
    """Select up to images_per_class samples per class from dataset.samples."""
    if not hasattr(dataset, "samples"):
        return []

    from collections import defaultdict

    class_samples: dict[int, List[Tuple[int, str, int]]] = defaultdict(list)
    for idx, (image_path, label) in enumerate(dataset.samples):
        if label < target_classes:
            class_samples[label].append((idx, image_path, label))

    selected_samples: List[Tuple[int, str, int]] = []
    for class_id in range(target_classes):
        if class_id in class_samples:
            class_images = class_samples[class_id][:images_per_class]
            selected_samples.extend(class_images)

    return selected_samples
