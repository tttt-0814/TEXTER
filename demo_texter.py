#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
from telnetlib import IP
import torch


from src.utils.load_models import setup_model
from src.overcomplete.sae import TopKSAE
from src.explanations.texter.texter import TEXTER
from src.explanations.generator.text_generator import LLMConceptGenerator
from src.utils.utils import (
    get_top_probs_and_labels,
    save_explainer_results,
)
from src.visualization.visualize import visualize_ranked_concepts
from src.datasets.load_datasets import load_dataset, select_samples_by_class


def parse_args():
    """
    Parse command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Experimental Code for Textual Explanations"
    )

    # Dataset arguments (ImageNet only)
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet18", "vit", "dino_vits8", "dino_resnet50"],
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation (default: cuda)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_demo",
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--concepts_data_path",
        type=str,
        default="Concepts/ImageNet/val",
        help="Path to concepts data directory",
    )
    parser.add_argument(
        "--aligner_model_path",
        type=str,
        default="pretrained_models/aligner",
        help="Path to the linear aligner model",
    )
    parser.add_argument(
        "--sae_model_path",
        type=str,
        default="pretrained_models/sae",
        help=(
            "Path to SAE checkpoint (.pth) or base directory. "
            "If directory is given, uses <dir>/<model_name>/10%/sae_model.pth."
        ),
    )
    parser.add_argument(
        "--target_classes",
        type=int,
        default=10,
        help="Number of ImageNet classes to sample.",
    )
    parser.add_argument(
        "--images_per_class",
        type=int,
        default=1,
        help="Number of images to process per class.",
    )
    parser.add_argument(
        "--topk_indices_for_ig",
        type=int,
        default=6,
        help="Top-k influential indices for integrated gradients in TEXTER.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Arguments: {vars(args)}")

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args to JSON
    args_file = output_dir / "args.json"
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Arguments saved to: {args_file}")

    # Setup model
    model = setup_model(args.model_name, pretrained=True, device=device)
    sae_model_path = Path(args.sae_model_path)
    if sae_model_path.suffix == ".pth":
        sae_checkpoint_path = sae_model_path
    else:
        sae_checkpoint_path = sae_model_path / args.model_name / "sae_model.pth"
    sae_checkpoint = torch.load(
        sae_checkpoint_path, map_location=device, weights_only=False
    )
    sae = TopKSAE(
        input_shape=sae_checkpoint["input_dim"],
        nb_concepts=sae_checkpoint["nb_concepts"],
        top_k=sae_checkpoint["top_k"],
        device=device,
    )
    sae.load_state_dict(sae_checkpoint["model_state_dict"])
    sae = sae.to(device)
    sae.eval()

    nle_generator = LLMConceptGenerator(
        model=model,
        model_name=args.model_name,
        device=device,
        openai_api_key="DUMMY_API_KEY",
    )
    nle_generator.load_linear_aligner(
        str(Path(args.aligner_model_path) / args.model_name / "linear_aligner.pth")
    )
    explainer = TEXTER(
        model=model,
        sae=sae,
        nle_generator=nle_generator,
        device=device,
        concepts_data_dir=args.concepts_data_path,
    )
    test_dataset, _, class_names = load_dataset("imagenet", args.data_root)
    print("Dataset: imagenet")
    print(f"Test samples: {len(test_dataset)}")

    selected_samples = select_samples_by_class(
        test_dataset, args.target_classes, args.images_per_class
    )
    if class_names is None:
        print("Class names not found, using class indices")

    print(
        f"Processing {len(selected_samples)} images "
        f"({args.target_classes} classes x up to {args.images_per_class} images/class)"
    )

    for idx, (_, image_path, label) in enumerate(selected_samples):
        image = test_dataset.loader(image_path)
        if test_dataset.transform:
            image = test_dataset.transform(image)
        images = image.unsqueeze(0)
        labels = torch.tensor([label])
        filename = Path(image_path).name

        top_probs, pred_name = get_top_probs_and_labels(
            images=images, model=model, class_names=class_names, device=device
        )

        results = explainer(
            images=images,
            gt_labels=labels,
            image_filenames=[filename],
            topk_indices_for_ig=args.topk_indices_for_ig,
            use_sae=True,
        )
        image_result = results[0]
        ranked_texts = image_result["sorted_concepts"]

        pred_class_name = pred_name
        vis_dir = (
            output_dir
            / "explanations"
            / Path(filename).stem
            / pred_class_name
            / "visualizations"
        )
        vis_dir.mkdir(parents=True, exist_ok=True)
        pred_prob = float(top_probs[0].item())
        gt_label = int(labels[0].item())
        gt_class_name = (
            class_names[gt_label]
            if class_names and gt_label < len(class_names)
            else str(gt_label)
        )
        compact_result = {
            "image_filename": filename,
            "gt_label": gt_label,
            "gt_class_name": gt_class_name,
            "pred_class_name": pred_class_name,
            "pred_prob": pred_prob,
            "predicted_label": int(image_result.get("predicted_label", -1)),
            "ranked_texts": ranked_texts,
            "num_concepts_total": len(ranked_texts),
        }
        save_explainer_results(
            results=compact_result,
            output_dir=args.output_dir,
            model_name=args.model_name,
            image_name=Path(filename).stem,
            target_class_name=pred_class_name,
        )
        concept_scores = ranked_texts[:10]
        out_path = vis_dir / f"{idx:05d}_{filename}.png"
        maco_vis = image_result.get("maco_visualization", {})
        visualize_ranked_concepts(
            image=image,
            pred_label=pred_name,
            pred_prob=pred_prob,
            ranked_concepts=concept_scores,
            save_path=str(out_path),
            maco_image=maco_vis.get("generated_image"),
            maco_alpha=maco_vis.get("alpha1"),
            title=filename,
            topk=10,
        )


if __name__ == "__main__":
    main()
