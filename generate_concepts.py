from pathlib import Path
import argparse
import json


import torch


from src.utils.load_models import setup_model
from src.datasets.load_datasets import load_dataset, select_samples_by_class
from src.explanations.generator.text_generator import (
    LLMConceptGenerator,
)


def parse_args():
    """
    Parse command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Concept-based Neural Language Explanation using SAE weights"
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
        "--batch_size",
        type=int,
        default=1,
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
        default="Concepts/ImageNet/val",
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--num_concepts_llm",
        type=int,
        default=100,
        help="Number of concepts to generate using LLM",
    )
    parser.add_argument(
        "--num_concepts_vlm",
        type=int,
        default=30,
        help="Number of concepts to generate using VLM",
    )
    parser.add_argument(
        "--target_classes",
        type=int,
        default=10,
        help="Number of classes to process from ImageNet val",
    )
    parser.add_argument(
        "--images_per_class",
        type=int,
        default=1,
        help="Number of images to process per class",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable",
    )

    args = parser.parse_args()

    # Save arguments to JSON file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args to JSON
    args_file = output_dir / "args.json"
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Arguments saved to: {args_file}")

    return args


def main():
    """
    Main execution function
    """
    # Parse command line arguments
    args = parse_args()

    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Arguments: {vars(args)}")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset, _, class_names = load_dataset("imagenet", args.data_root)

    print("Dataset: imagenet")
    print(f"Test samples: {len(test_dataset)}")

    model = setup_model(args.model_name, pretrained=True, device=device)
    nle_generator = LLMConceptGenerator(
        model=model, model_name=args.model_name, openai_api_key=args.openai_api_key
    )
    selected_samples = select_samples_by_class(
        test_dataset, args.target_classes, args.images_per_class
    )

    print(
        f"Processing {len(selected_samples)} images "
        f"({args.target_classes} classes x up to {args.images_per_class} images/class)"
    )

    for _, image_path, label in selected_samples:
        image = test_dataset.loader(image_path)
        if test_dataset.transform:
            image = test_dataset.transform(image)
        original_image = image.to(device)
        image_filename = Path(image_path).name
        label_name = class_names[label] if class_names else str(label)

        if model.has_normalizer:
            image = model.get_normalizer(original_image)[None, ...]
        else:
            image = original_image[None, ...]

        pred_label = (
            model.classifier(model.forward_features(image).squeeze(-1).squeeze(-1))
            .argmax()
            .item()
        )
        pred_label_name = class_names[pred_label] if class_names else str(pred_label)

        concepts_class_json_dir = output_dir / "per_class" / pred_label_name
        concepts_class_json_path = (
            concepts_class_json_dir / f"concepts_{args.num_concepts_llm}.json"
        )
        concepts_per_image_dir = (
            output_dir / "per_image" / image_filename / pred_label_name
        )
        concepts_per_image_vlm_json_path = (
            concepts_per_image_dir / f"concepts_vlm_{args.num_concepts_vlm}.json"
        )

        if (
            concepts_class_json_path.exists()
            and concepts_per_image_vlm_json_path.exists()
        ):
            print(f"Skipping {image_filename} because it already exists")
            continue
        else:
            if concepts_class_json_path.exists():
                concepts_llm = json.load(open(concepts_class_json_path, "r"))
                print(f"Loaded concepts from: {concepts_class_json_path}")
            else:
                concepts_class_json_dir.mkdir(parents=True, exist_ok=True)
                concepts_llm = (
                    nle_generator.concept_extractor.generate_concepts_from_llm(
                        class_name=pred_label_name,
                        num_concepts=args.num_concepts_llm,
                    )
                )
                with open(concepts_class_json_path, "w") as f:
                    json.dump(concepts_llm, f, indent=2)
                print(f"Saved concepts to: {concepts_class_json_path}")

            concepts_per_image_dir.mkdir(parents=True, exist_ok=True)
            concepts_vlm = nle_generator.concept_extractor.generate_concepts_from_vlm(
                image=original_image,
                class_name=pred_label_name,
                num_concepts=args.num_concepts_vlm,
            )
            with open(concepts_per_image_vlm_json_path, "w") as f:
                json.dump(concepts_vlm, f, indent=2)
            print(f"Saved concepts to: {concepts_per_image_vlm_json_path}")

            # Combine and deduplicate concepts
            all_concepts = concepts_llm + concepts_vlm
            unique_concepts = []
            seen_concepts = set()
            for concept in all_concepts:
                # Handle both string and dict formats
                if isinstance(concept, str):
                    concept_text = concept.lower().strip()
                else:
                    concept_text = concept.get("concept", "").lower().strip()

                # Skip concepts that contain class names
                if (
                    concept_text
                    and concept_text not in seen_concepts
                    and not any(
                        class_name.lower() in concept_text
                        for class_name in [pred_label_name, label_name]
                    )
                ):
                    unique_concepts.append(concept)
                    seen_concepts.add(concept_text)

            concepts_per_image_all_json_path = (
                concepts_per_image_dir / "concepts_all.json"
            )
            with open(concepts_per_image_all_json_path, "w") as f:
                json.dump(unique_concepts, f, indent=2)
            print(
                f"Saved {len(unique_concepts)} unique concepts to: {concepts_per_image_all_json_path}"
            )


if __name__ == "__main__":
    main()
