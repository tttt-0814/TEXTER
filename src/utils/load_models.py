import torch
import torchvision
import torch.nn as nn
from pathlib import Path
from src.explanations.text_to_concept.linear_aligner import LinearAligner
from typing import Optional


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_linear_aligner(model_path: str) -> LinearAligner:
    linear_aligner = LinearAligner()
    linear_aligner.load_W(str(_resolve_project_path(model_path)))

    return linear_aligner


def get_last_conv_layer_name(model: nn.Module) -> Optional[str]:
    """
    Get the name of the last convolutional layer in the model

    Args:
        model: Target model

    Returns:
        Name of the last convolutional layer, None if not found
    """
    last_conv_name = None

    # Explore all named modules in the model
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            last_conv_name = name

    return last_conv_name


def get_last_conv_layer_name_by_model_type(model: nn.Module) -> Optional[str]:
    """
    Get the name of the last convolutional layer based on model structure

    Args:
        model: Target model

    Returns:
        Name of the last convolutional layer, None if not found
    """
    # Get model class name to determine architecture type
    model_class_name = model.__class__.__name__.lower()

    # For ResNet series - check if model has ResNet-like structure
    if "resnet" in model_class_name or any(
        "layer4" in name for name, _ in model.named_modules()
    ):
        # For ResNet, the last conv layer is usually the last conv in layer4
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and "layer4" in name:
                last_conv_name = name
        return last_conv_name if "last_conv_name" in locals() else None

    # For ViT series - check if model has Vision Transformer structure
    elif "vit" in model_class_name or "transformer" in model_class_name:
        # ViT usually doesn't have Conv layers, so return None
        return None

    # For other models, perform general search
    else:
        return get_last_conv_layer_name(model)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def setup_resnet(model: nn.Module, device: str = "cpu") -> torch.nn.Module:
    encoder = torch.nn.Sequential(*list(model.children())[:-1])

    # Add forward_features method for feature extraction
    model.forward_features = lambda x: encoder(x)
    model.classifier = lambda x: model.fc(x)

    # Add normalizer
    model.get_normalizer = torchvision.transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    model.has_normalizer = True

    # Move to device
    model = model.to(device)
    model.eval()

    return model


def setup_resnet50(pretrained: bool = True, device: str = "cpu") -> torch.nn.Module:
    """
    Setup ResNet50 model with required attributes for TextToConcept.

    Args:
        pretrained: Whether to load pretrained weights
        device: Device to load the model on

    Returns:
        Configured ResNet50 model with additional attributes
    """
    model = torchvision.models.resnet50(pretrained=pretrained)

    # Create feature encoder (remove classification head)
    encoder = torch.nn.Sequential(*list(model.children())[:-1])

    # Add forward_features method for feature extraction
    model.forward_features = lambda x: encoder(x)
    model.classifier = lambda x: model.fc(x)

    # Add normalizer
    model.get_normalizer = torchvision.transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    model.has_normalizer = True

    # Move to device
    model = model.to(device)
    model.eval()

    return model


def setup_resnet18(pretrained: bool = True, device: str = "cpu") -> torch.nn.Module:
    """
    Setup ResNet18 model with required attributes for TextToConcept.

    Args:
        pretrained: Whether to load pretrained weights
        device: Device to load the model on

    Returns:
        Configured ResNet18 model with additional attributes
    """
    model = torchvision.models.resnet18(pretrained=pretrained)

    # Create feature encoder (remove classification head)
    encoder = torch.nn.Sequential(*list(model.children())[:-1])

    # Add forward_features method for feature extraction
    model.forward_features = lambda x: encoder(x)
    model.classifier = lambda x: model.fc(x)

    # Add normalizer
    model.get_normalizer = torchvision.transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    model.has_normalizer = True

    # Move to device
    model = model.to(device)
    model.eval()

    return model


def setup_vit(pretrained: bool = True, device: str = "cpu") -> torch.nn.Module:
    """
    Setup ViT model with required attributes for TextToConcept.

    Args:
        pretrained: Whether to load pretrained weights
        device: Device to load the model on

    Returns:
        Configured ViT model with additional attributes
    """

    def forward_features(model, x):
        x = model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x

    model = torchvision.models.vit_b_16(pretrained=pretrained)

    # Add forward_features method for feature extraction
    model.forward_features = lambda x: forward_features(model, x)

    # Keep the original classifier (head)
    model.classifier = lambda x: model.heads.head(x)

    # Add normalizer
    model.get_normalizer = torchvision.transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    model.has_normalizer = True

    # Move to device
    model = model.to(device)
    model.eval()

    return model


def setup_dino_vit(
    model: nn.Module,
    device: str = "cpu",
    classification_weights_path: str = "pretrained_models/linear_classification_imagenet/dino_resnet50/dino_resnet50_linearweights.pth",
    num_layers: int = 4,
    linear_layer_dim: int = 1536,
) -> torch.nn.Module:
    def forward_features(model, x):
        intermediate_output = model.get_intermediate_layers(x, num_layers)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

        return output

    model.forward_features = lambda x: forward_features(model, x)

    # Load classification weights
    ckpt = torch.load(
        _resolve_project_path(classification_weights_path), map_location="cpu"
    )
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    # Convert state keys from 'module.linear.*' to 'linear.*'
    converted_state = {}
    for key, value in state.items():
        if key.startswith("module.linear."):
            new_key = key.replace("module.linear.", "linear.")
            converted_state[new_key] = value
        else:
            converted_state[key] = value

    head = LinearClassifier(linear_layer_dim, num_labels=1000)
    head.load_state_dict(converted_state, strict=True)
    head = head.to(device)

    # Add classifier method (DINO ViT doesn't have classification head)
    model.head = head
    model.classifier = lambda x: head(x)

    model.has_normalizer = True
    model.get_normalizer = torchvision.transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD
    )

    # Move to device
    model = model.to(device)
    model.eval()

    return model


def setup_dino_vits8(
    device: str = "cpu",
    classification_weights_path: str = "pretrained_models/linear_classification_imagenet/dino_vits8/dino_deitsmall8_linearweights.pth",
    num_layers: int = 4,
    linear_layer_dim: int = 1536,
) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
    model = setup_dino_vit(
        model, device, classification_weights_path, num_layers, linear_layer_dim
    )

    return model


def setup_dino_resnet50(
    device: str = "cpu",
    classification_weights_path: str = "pretrained_models/linear_classification_imagenet/dino_resnet50/dino_resnet50_linearweights.pth",
    linear_layer_dim: int = 2048,
) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    model = setup_resnet(model, device)

    # Load classification weights
    ckpt = torch.load(
        _resolve_project_path(classification_weights_path), map_location="cpu"
    )
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    # Convert state keys from 'module.linear.*' to 'linear.*'
    converted_state = {}
    for key, value in state.items():
        if key.startswith("module.linear."):
            new_key = key.replace("module.linear.", "linear.")
            converted_state[new_key] = value
        else:
            converted_state[key] = value

    head = LinearClassifier(linear_layer_dim, num_labels=1000)
    head.load_state_dict(converted_state, strict=True)
    head = head.to(device)

    # Add classifier method
    model.head = head
    model.classifier = lambda x: head(x)

    # Add normalizer (DINO ResNet uses ImageNet normalization)
    model.get_normalizer = torchvision.transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    model.has_normalizer = True

    return model


def setup_model(
    model_name: str = "resnet50", pretrained: bool = True, device: str = "cpu"
) -> torch.nn.Module:
    """
    Setup model based on the specified model name.

    Args:
        model_name: Name of the model to load ("resnet50", "vit", "vit_b_16")
        pretrained: Whether to load pretrained weights
        device: Device to load the model on

    Returns:
        Configured model with required attributes

    Raises:
        ValueError: If unsupported model name is provided
    """
    model_name = model_name.lower()

    print(f"Setup model: {model_name}")
    if model_name in ["resnet50"]:
        return setup_resnet50(pretrained=pretrained, device=device)
    elif model_name in ["resnet18"]:
        return setup_resnet18(pretrained=pretrained, device=device)
    elif model_name in ["vit", "vit_b_16", "vision_transformer"]:
        return setup_vit(pretrained=pretrained, device=device)
    elif model_name in ["dino_vits8"]:
        return setup_dino_vits8(device=device)
    elif model_name in ["dino_resnet50"]:
        return setup_dino_resnet50(device=device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. ")
