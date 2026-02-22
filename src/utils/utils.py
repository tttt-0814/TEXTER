import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
from typing import Optional, List, Union, Dict, Any


def load_concepts(
    image_filename: str,
    concepts_data_dir: Union[str, Path],
    class_name: Optional[str] = None,
    concept_file: str = "concepts_all.json",
) -> Optional[List[Union[str, Dict[str, Any]]]]:
    """
    Load concepts from file corresponding to the given image filename.

    Args:
        image_filename: Image filename (e.g., "ILSVRC2012_val_00000001.JPEG")
        concepts_data_dir: Root directory for concepts data
        class_name: Class name (optional). If not specified, automatically searches for concepts
        concept_file: Name of the concept file to load (default: "concepts_all.json")

    Returns:
        List of concepts. Returns None if file is not found.

    Examples:
        >>> concepts = load_concepts(
        ...     "ILSVRC2012_val_00000001.JPEG",
        ...     "Concepts_1000/ImageNet/val",
        ...     class_name="n01440764"
        ... )
    """
    concepts_data_dir = Path(concepts_data_dir)

    # Path to per_image directory
    per_image_dir = concepts_data_dir / "per_image" / image_filename

    if not per_image_dir.exists():
        return None

    # If class_name is specified
    if class_name is not None:
        concept_path = per_image_dir / class_name / concept_file
        if concept_path.exists():
            try:
                with open(concept_path, "r") as f:
                    print(f"Loaded concepts from: {concept_path}")
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading concepts from {concept_path}: {e}")
                return None
        return None

    # If class_name is not specified, search for the first concept file found
    # Explore subdirectories under per_image_dir
    if per_image_dir.is_dir():
        for subdir in per_image_dir.iterdir():
            if subdir.is_dir():
                concept_path = subdir / concept_file
                if concept_path.exists():
                    try:
                        with open(concept_path, "r") as f:
                            print(f"Loaded concepts from: {concept_path}")
                            return json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Error loading concepts from {concept_path}: {e}")
                        continue

    return None


def _sanitize_path_segment(segment: str) -> str:
    return str(segment).replace("/", "_").replace("\\", "_")


def _slice_results_for_index(results: Any, index: int) -> Any:
    if torch.is_tensor(results):
        if results.ndim > 0 and results.size(0) > index:
            return results[index]
        return results
    if isinstance(results, list):
        if len(results) > index:
            return results[index]
        return results
    if isinstance(results, dict):
        return {k: _slice_results_for_index(v, index) for k, v in results.items()}
    return results


def _prepare_results_for_save(results: Any) -> Any:
    if torch.is_tensor(results):
        return results.detach().cpu()
    if isinstance(results, dict):
        return {k: _prepare_results_for_save(v) for k, v in results.items()}
    if isinstance(results, list):
        return [_prepare_results_for_save(v) for v in results]
    if isinstance(results, tuple):
        return tuple(_prepare_results_for_save(v) for v in results)
    return results


def _prepare_results_for_json(results: Any) -> Any:
    if torch.is_tensor(results):
        return results.detach().cpu().tolist()
    if isinstance(results, np.ndarray):
        return results.tolist()
    if isinstance(results, np.generic):
        return results.item()
    if isinstance(results, dict):
        return {k: _prepare_results_for_json(v) for k, v in results.items()}
    if isinstance(results, list):
        return [_prepare_results_for_json(v) for v in results]
    if isinstance(results, tuple):
        return [_prepare_results_for_json(v) for v in results]
    return results


def save_explainer_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    model_name: str,
    image_name: str,
    target_class_name: str,
    index: Optional[int] = None,
) -> Path:
    save_dir = (
        Path(output_dir)
        / _sanitize_path_segment(model_name)
        / "explanations"
        / _sanitize_path_segment(image_name)
        / _sanitize_path_segment(target_class_name)
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    save_results = results
    if index is not None:
        save_results = _slice_results_for_index(results, index)
    save_results = _prepare_results_for_save(save_results)

    json_path = save_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(
            _prepare_results_for_json(save_results),
            f,
            ensure_ascii=True,
            indent=2,
        )
    return json_path


def get_top_probs_and_labels(
    images: torch.Tensor,
    model: torch.nn.Module,
    class_names: Optional[List[str]],
    device: str,
) -> tuple[torch.Tensor, str]:
    with torch.no_grad():
        images_device = images.to(device)
        if getattr(model, "has_normalizer", False):
            images_device = model.get_normalizer(images_device)

        features = model.forward_features(images_device)
        class_logits = model.classifier(features.flatten(1))
        probs = F.softmax(class_logits, dim=-1)
        top_probs, top_idxs = probs.max(dim=-1)

    pred_idx = int(top_idxs[0].item())
    pred_name = (
        class_names[pred_idx]
        if class_names and pred_idx < len(class_names)
        else str(pred_idx)
    )
    return top_probs, pred_name


def get_top_probs_and_lables(
    images: torch.Tensor,
    model: torch.nn.Module,
    class_names: Optional[List[str]],
    device: str,
) -> tuple[torch.Tensor, str]:
    return get_top_probs_and_labels(
        images=images, model=model, class_names=class_names, device=device
    )
