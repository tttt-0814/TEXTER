from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from src.horama import maco
from src.overcomplete.sae import TopKSAE
from src.horama.utils import preprocess_for_maco_image
from src.utils.class_labels import IMAGENET_CLASSES
from src.explanations.generator.text_generator import LLMConceptGenerator


class LambdaModule(nn.Module):
    """Custom Lambda module since torch.nn.Lambda doesn't exist."""

    def __init__(self, func):
        super(LambdaModule, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class TEXTER(BaseModel):
    """TEXTER: SAE-based Concept Neural Language Explanation class."""

    model: torch.nn.Module = Field(..., description="Vision model to explain")
    sae: TopKSAE = Field(..., description="SAE model for visual model")
    nle_generator: LLMConceptGenerator = Field(None, description="NLE generator model")
    device: str = Field(default="cuda", description="Device to use for computation")
    class_names: List[str] = Field(
        default_factory=lambda: IMAGENET_CLASSES, description="Class names for data"
    )
    past_data_dir: str | Path = Field(
        default="path_to_past_data", description="Path to past data directory"
    )
    concepts_data_dir: str | Path = Field(
        default="path_to_concepts_data", description="Path to concepts data directory"
    )

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any):
        self.model.eval()
        self.sae.eval()
        self.past_data_dir = Path(self.past_data_dir) / "explanations"

    def __call__(
        self,
        images: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
        num_concepts_llm: int = 100,
        num_concepts_vlm: int = 50,
        image_filenames: Optional[List[str]] = None,
        use_sae: bool = True,
        topk_indices_for_ig: int = 6,
        target_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # 1. Identify influential concepts in SAE and generate MACO images
        if use_sae:
            topk_indices = topk_indices_for_ig
        else:
            topk_indices = topk_indices_for_ig

        results = self.identify_influential_concepts(
            images,
            image_filenames=image_filenames,
            gt_labels=gt_labels,
            target_labels=target_labels,
            topk_indices=topk_indices,
            use_sae=use_sae,
        )

        # Add ground truth labels and image filenames to the results
        for i, image_result in enumerate(results):
            image_result["gt_label"] = gt_labels[i]
            image_result["image_filename"] = image_filenames[i]

        # 2. Generate NLE for MACO images
        generated_images = []
        predicted_labels = []
        for image_result in results:
            generated_image = image_result["maco_visualization"]["generated_image"]
            alpha1 = image_result["maco_visualization"]["alpha1"]
            processed_image = preprocess_for_maco_image(generated_image, alpha1)
            generated_images.append(processed_image)
            predicted_labels.append(image_result["predicted_label"])
        maco_tensors = torch.stack(generated_images, dim=0)

        list_sorted_concepts, list_similarities, list_best_idx = (
            self.concepts_for_maco_images(
                images,
                maco_tensors,
                predicted_labels,
                num_concepts_llm=num_concepts_llm,
                num_concepts_vlm=num_concepts_vlm,
                image_filenames=image_filenames,
            )
        )
        # Add concept results to the SAE results
        for i, image_result in enumerate(results):
            if i < len(list_sorted_concepts):
                image_result["sorted_concepts"] = list_sorted_concepts[i]
                image_result["similarities"] = (
                    list_similarities[i].tolist()
                    if hasattr(list_similarities[i], "tolist")
                    else list_similarities[i]
                )
                image_result["best_idx"] = list_best_idx[i]

        return results

    def concepts_for_maco_images(
        self,
        original_images: torch.Tensor,
        maco_images: torch.Tensor,
        predicted_labels: List[int],
        num_concepts_llm: int = 100,
        num_concepts_vlm: int = 50,
        image_filenames: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate concepts for MACO images"""

        batch_size = maco_images.shape[0]
        list_sorted_concepts = []
        list_similarities = []
        list_best_idx = []

        for i in range(batch_size):
            original_image = original_images[i][None, :, :, :]
            maco_image = maco_images[i]
            predicted_label = predicted_labels[i]
            image_filename = image_filenames[i] if image_filenames else f"{i:07d}"

            concepts_result_path = (
                Path(self.concepts_data_dir)
                / "per_image"
                / f"{image_filename}"
                / f"{self.class_names[predicted_label]}"
                / f"concepts_all.json"
            )

            if concepts_result_path.exists():
                print(
                    f"Loading existing concepts for image {i}, class {predicted_label}"
                )
                with open(concepts_result_path, "r") as f:
                    all_generated_concepts = json.load(f)
            else:
                print(f"Generating new concepts for image {i}, class {predicted_label}")
                llm_generated_cocepts_for_predicted_label = (
                    self.nle_generator.concept_extractor.generate_concepts_from_llm(
                        class_name=self.class_names[predicted_label],
                        num_concepts=num_concepts_llm,
                    )
                )
                vlm_generated_cocepts_for_predicted_label = (
                    self.nle_generator.concept_extractor.generate_concepts_from_vlm(
                        image=original_image,
                        class_name=self.class_names[predicted_label],
                        num_concepts=num_concepts_vlm,
                    )
                )
                all_generated_concepts = (
                    llm_generated_cocepts_for_predicted_label
                    + vlm_generated_cocepts_for_predicted_label
                )

            aligned_image_features = (
                self.nle_generator.get_img_feature_from_aligned_reps(
                    maco_image, normalize=False
                )
            )
            aligned_image_features = (
                aligned_image_features
                / aligned_image_features.norm(dim=-1, keepdim=True)
            )

            sorted_concept, similarities, best_idx = (
                self.nle_generator.compute_concept_similarity_for_maco_features(
                    maco_features=aligned_image_features,
                    candidate_concepts=all_generated_concepts,
                )
            )

            list_sorted_concepts.append(sorted_concept)
            list_similarities.append(similarities)
            list_best_idx.append(best_idx)

        return list_sorted_concepts, list_similarities, list_best_idx

    def identify_influential_concepts(
        self,
        images: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        image_filenames: Optional[List[str]] = None,
        topk_indices: int = 3,
        use_sae: bool = True,
        num_maco_iter: int = 512,
    ) -> Dict[str, Any]:
        # Process single batch
        images = self.model.get_normalizer(images.to(self.device))

        pred_forward_features = self.model.forward_features(images)
        pred_forward_features = pred_forward_features.view(
            pred_forward_features.size(0), -1
        )
        pred_logits = self.model.classifier(pred_forward_features)

        if target_labels is not None:
            pred_labels = target_labels
        else:
            pred_labels = pred_logits.argmax(dim=1)

        pre_codes, topk_codes, sae_reconstructed = self.sae(pred_forward_features)

        batch_size = pre_codes.shape[0]
        batch_results = []
        for i in range(batch_size):
            pre_code_i = pre_codes[i]
            topk_code_i = topk_codes[i]
            sae_reconstructed_code_i = sae_reconstructed[i]
            original_pred_label_i = pred_labels[i]
            pred_forward_features_i = pred_forward_features[i]

            print(
                f"Identify influential concepts and generate maco images for label: {original_pred_label_i}"
            )

            # Get influential index
            if use_sae:
                influence_result = self.get_influential_idx_for_sae(
                    topk_code_i,
                    sae_reconstructed_code_i,
                    original_pred_label_i,
                    mode="IG",
                )
            else:
                influence_result = self.get_influential_idx_for_forward_features(
                    pred_forward_features_i, original_pred_label_i
                )

            # Get top-k influential indices from influence_scores
            influence_scores = influence_result["influence_scores"]
            # Sort by influence score in descending order and get top-k
            sorted_indices = sorted(
                influence_scores.keys(),
                key=lambda k: influence_scores[k],
                reverse=True,
            )

            top_k_indices = sorted_indices[: min(topk_indices, len(sorted_indices))]
            # Check if results already exist
            pred_label = original_pred_label_i.item()
            image_filename = image_filenames[i] if image_filenames else f"{i:07d}"

            maco_result_path = (
                Path(self.past_data_dir)
                / self.class_names[gt_labels[i]]
                / "data"
                / "maco_images"
                / f"{image_filename}"
                / f"{pred_label}"
                / f"maco.npz"
            )
            if maco_result_path.exists():
                print(
                    f"Loading existing maco visualization for concept {top_k_indices}"
                )
                # Load existing results
                existing_data = np.load(maco_result_path)
                image1 = torch.from_numpy(existing_data["maco_image"])
                alpha1 = torch.from_numpy(existing_data["alpha1"])
            else:
                while True:
                    objective_model = self.get_objective_model(
                        top_k_indices, use_sae=use_sae
                    )
                    # Generate visualization using maco
                    print(f"Generating maco visualization for concept {top_k_indices}")
                    image1, alpha1 = maco(
                        objective_model, total_steps=num_maco_iter, device="cuda"
                    )
                    if alpha1.max() > 0:
                        break
                    else:
                        print(
                            f"Alpha1 is 0, increasing topk_indices to {topk_indices + 1}"
                        )
                        topk_indices += 1
                        top_k_indices = sorted_indices[
                            : min(topk_indices, len(sorted_indices))
                        ]

            # Compile results for this image
            image_result = {
                "predicted_label": original_pred_label_i.item(),
                "predicted_logits": pred_logits[i].detach().cpu().numpy(),
                "pre_codes": pre_code_i.detach().cpu().numpy(),
                "topk_codes": topk_code_i.detach().cpu().numpy(),
                "sae_reconstructed": sae_reconstructed_code_i.detach().cpu().numpy(),
                "influence_analysis": influence_result,
                "top_k_influential_indices": top_k_indices,
                "maco_visualization": {
                    "generated_image": image1,
                    "alpha1": alpha1,
                },
            }

            batch_results.append(image_result)

        return batch_results

    def get_objective_model(
        self, influential_indices: List[int], use_sae: bool = True
    ) -> nn.Sequential:
        """
        Create objective model for SAE neuron visualization.

        Args:
            influential_indices: Index or list of indices of the most influential SAE neurons

        Returns:
            torch.nn.Sequential: Objective model for visualization
        """

        if isinstance(influential_indices, int):
            if use_sae:
                # Single neuron case
                objective_model = nn.Sequential(
                    # Step 0: Normalize input
                    LambdaModule(lambda x: self.model.get_normalizer(x)),
                    # Step 1: Extract features
                    LambdaModule(
                        lambda x: self.model.forward_features(x).squeeze(-1).squeeze(-1)
                    ),
                    # Step 2: SAE encode to get topk_codes
                    LambdaModule(
                        lambda x: self.sae.encode(x)[1]
                    ),  # [1] gets topk_codes
                    # Step 3: Get activation of specific neuron
                    LambdaModule(lambda x: torch.mean(x[:, influential_indices])),
                )
            else:
                objective_model = nn.Sequential(
                    # Step 0: Normalize input
                    LambdaModule(lambda x: self.model.get_normalizer(x)),
                    # Step 1: Extract features
                    LambdaModule(
                        lambda x: self.model.forward_features(x).squeeze(-1).squeeze(-1)
                    ),
                    # Step 2: Get activation of specific neuron
                    LambdaModule(lambda x: torch.mean(x[:, influential_indices])),
                )
        elif isinstance(influential_indices, list) and len(influential_indices) > 1:
            # Multiple neurons case
            indices_list = influential_indices
            if use_sae:
                objective_model = nn.Sequential(
                    # Step 0: Normalize input
                    LambdaModule(lambda x: self.model.get_normalizer(x)),
                    # Step 1: Extract features
                    LambdaModule(
                        lambda x: self.model.forward_features(x).squeeze(-1).squeeze(-1)
                    ),
                    # Step 2: SAE encode to get topk_codes
                    LambdaModule(
                        lambda x: self.sae.encode(x)[1]
                    ),  # [1] gets topk_codes
                    # Step 3: Sum activations of multiple neurons (each neuron's mean activation)
                    LambdaModule(
                        lambda x: torch.sum(
                            torch.stack([torch.mean(x[:, idx]) for idx in indices_list])
                        )
                    ),
                )
            else:
                objective_model = nn.Sequential(
                    # Step 0: Normalize input
                    LambdaModule(lambda x: self.model.get_normalizer(x)),
                    # Step 1: Extract features
                    LambdaModule(
                        lambda x: self.model.forward_features(x).squeeze(-1).squeeze(-1)
                    ),
                    # Step 2: Sum activations of multiple neurons (each neuron's mean activation)
                    LambdaModule(
                        lambda x: torch.sum(
                            torch.stack([torch.mean(x[:, idx]) for idx in indices_list])
                        )
                    ),
                )
        elif isinstance(influential_indices, list) and len(influential_indices) == 1:
            # Single element list -> treat as single index
            single_index = influential_indices[0]
            if use_sae:
                objective_model = nn.Sequential(
                    LambdaModule(lambda x: self.model.get_normalizer(x)),
                    LambdaModule(
                        lambda x: self.model.forward_features(x).squeeze(-1).squeeze(-1)
                    ),
                    LambdaModule(lambda x: self.sae.encode(x)[1]),
                    LambdaModule(lambda x: torch.mean(x[:, single_index])),
                )
            else:
                objective_model = nn.Sequential(
                    LambdaModule(lambda x: self.model.get_normalizer(x)),
                    LambdaModule(
                        lambda x: self.model.forward_features(x).squeeze(-1).squeeze(-1)
                    ),
                    LambdaModule(lambda x: torch.mean(x[:, single_index])),
                )
        elif isinstance(influential_indices, list) and len(influential_indices) == 0:
            raise ValueError(
                "influential_indices is an empty list; cannot build objective model"
            )
        else:
            raise TypeError(
                f"Unsupported type for influential_indices: {type(influential_indices)}"
            )

        return objective_model

    def get_influential_idx_for_sae(
        self,
        topk_code_i: torch.Tensor,
        sae_reconstructed_code_i: torch.Tensor,
        original_pred_label_i: torch.Tensor,
        mode: str = "pertubation",
    ) -> Dict[str, Any]:
        """
        Analyze SAE neuron influence for a single image.

        Args:
            topk_code_i: Top-k codes for a single image
            sae_reconstructed_code_i: SAE reconstructed features for a single image
            original_pred_label_i: Original prediction label

        Returns:
            Dict containing influence_scores, influence_details, and most_influential_idx
        """
        if mode == "pertubation":
            # Create a mask for non-zero elements
            non_zero_mask = topk_code_i != 0

            # Get indices of non-zero elements
            non_zero_indices = (
                torch.nonzero(non_zero_mask, as_tuple=True)[0].cpu().numpy().tolist()
            )

            # Calculate original prediction score
            reconstructed_logits = self.model.classifier(
                sae_reconstructed_code_i.unsqueeze(0)
            ).squeeze(0)
            target_logits = reconstructed_logits[original_pred_label_i]

            # Find the most influential index by testing each non-zero index
            influence_scores = {}
            influence_details = {}  # Record detailed information for each index

            for idx in non_zero_indices:
                # Create a modified copy of topk_code_i
                modified_topk_code = topk_code_i.clone()

                # Record original value at this index
                original_value = topk_code_i[idx].item()

                # Set the specific index to zero to test influence
                modified_topk_code[idx] = 0

                # Reconstruct using SAE decoder
                modified_reconstructed = self.sae.decode(
                    modified_topk_code.unsqueeze(0)
                ).squeeze(0)

                # Calculate new prediction score
                modified_logits = self.model.classifier(
                    modified_reconstructed.unsqueeze(0)
                ).squeeze(0)
                target_modified_logit = modified_logits[original_pred_label_i]

                # Calculate score change (absolute difference)
                score_change = abs(target_logits - target_modified_logit)
                influence_scores[idx] = score_change

                # Record detailed information
                influence_details[idx] = {
                    "original_value": original_value,
                    "score_change": score_change.item(),
                    "target_logits": target_logits.item(),
                    "target_modified_logit": target_modified_logit.item(),
                }

            # Find the most influential index
            most_influential_idx = (
                max(influence_scores.keys(), key=lambda k: influence_scores[k])
                if influence_scores
                else None
            )
        elif mode == "IG":
            # Create a wrapper function that takes SAE codes as input and returns class logits
            def sae_to_logits(sae_codes):
                """
                Wrapper function: SAE codes -> reconstructed features -> class logits
                Args:
                    sae_codes: [batch_size, num_concepts] SAE activation codes
                Returns:
                    logits: [batch_size, num_classes] class prediction logits
                """
                # Decode SAE codes to reconstructed features
                reconstructed_features = self.sae.decode(sae_codes)
                # Get class logits from reconstructed features
                logits = self.model.classifier(reconstructed_features)
                return logits

            # Prepare input for Integrated Gradients
            input_codes = topk_code_i.unsqueeze(
                0
            )  # Add batch dimension [1, num_concepts]
            target_class = original_pred_label_i.item()  # Convert to scalar

            # Create Integrated Gradients instance
            ig = IntegratedGradients(sae_to_logits)

            # Calculate attributions
            # Use zero baseline (all SAE codes set to 0)
            baseline = torch.zeros_like(input_codes)

            # Compute attributions for the target class
            attributions = ig.attribute(
                input_codes,
                baselines=baseline,
                target=target_class,
                n_steps=100,  # Number of steps for integration
            )

            # Get attribution scores for each SAE neuron
            attribution_scores = attributions.squeeze(0)  # Remove batch dimension

            # Find non-zero indices (active SAE neurons)
            non_zero_mask = topk_code_i != 0
            non_zero_indices = (
                torch.nonzero(non_zero_mask, as_tuple=True)[0].cpu().numpy().tolist()
            )

            # Check for significant attributions outside active neurons
            zero_mask = topk_code_i == 0
            zero_attributions = attribution_scores[zero_mask]
            significant_zero_mask = torch.abs(zero_attributions) > 1e-6

            if significant_zero_mask.any():
                zero_indices = torch.nonzero(zero_mask, as_tuple=True)[0][
                    significant_zero_mask
                ]
                print(
                    f"Found {len(zero_indices)} zero-code neurons with significant attributions"
                )

            # Create influence scores and details based on attributions
            influence_scores = {}
            influence_details = {}
            for idx in non_zero_indices:
                attribution_score = attribution_scores[idx].item()
                original_value = topk_code_i[idx].item()

                # Use absolute attribution as influence score
                influence_scores[idx] = abs(attribution_score)

                # Record detailed information
                influence_details[idx] = {
                    "original_value": original_value,
                    "attribution_score": attribution_score,
                    "abs_attribution": abs(attribution_score),
                }

            # Find the most influential index (highest absolute attribution)
            most_influential_idx = (
                max(influence_scores.keys(), key=lambda k: influence_scores[k])
                if influence_scores
                else None
            )

        return {
            "influence_scores": influence_scores,
            "influence_details": influence_details,
            "most_influential_idx": most_influential_idx,
        }

    def get_influential_idx_for_forward_features(
        self,
        forward_features: torch.Tensor,
        original_pred_label: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Use Integrated Gradients to find influential indices in forward features.

        Args:
            forward_features: Forward features tensor [num_features]
            original_pred_label: Original prediction label

        Returns:
            Dict containing influence_scores, influence_details, and most_influential_idx
        """

        # Create a wrapper function that takes forward features as input and returns class logits
        def features_to_logits(features):
            """
            Wrapper function: forward features -> class logits
            Args:
                features: [batch_size, num_features] forward features
            Returns:
                logits: [batch_size, num_classes] class prediction logits
            """
            # Get class logits from forward features
            logits = self.model.classifier(features)
            return logits

        # Prepare input for Integrated Gradients
        input_features = forward_features.unsqueeze(
            0
        )  # Add batch dimension [1, num_features]

        # Create baseline (zero tensor)
        baseline = torch.zeros_like(input_features)

        # Initialize Integrated Gradients
        ig = IntegratedGradients(features_to_logits)

        # Compute attributions
        attributions = ig.attribute(
            inputs=input_features,
            baselines=baseline,
            target=original_pred_label.item(),
            n_steps=100,
        )

        # Get attribution scores for the target class
        attribution_scores = attributions.squeeze(0)  # Remove batch dimension

        # Create influence scores based on attributions
        influence_scores = {}
        influence_details = {}

        for idx in range(len(attribution_scores)):
            attribution_score = attribution_scores[idx].item()
            original_value = forward_features[idx].item()

            # Use absolute attribution as influence score
            influence_scores[idx] = abs(attribution_score)

            # Record detailed information
            influence_details[idx] = {
                "original_value": original_value,
                "attribution_score": attribution_score,
                "abs_attribution": abs(attribution_score),
            }

        # Find the most influential index (highest absolute attribution)
        most_influential_idx = (
            max(influence_scores.keys(), key=lambda k: influence_scores[k])
            if influence_scores
            else None
        )

        return {
            "influence_scores": influence_scores,
            "influence_details": influence_details,
            "most_influential_idx": most_influential_idx,
        }

    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_dir: Path,
        class_names: List[str] = IMAGENET_CLASSES,
    ) -> None:
        for i, image_result in enumerate(results):
            _class_name = class_names[image_result["gt_label"].item()]
            data_dir_maco = output_dir / _class_name / "data" / "maco_images"
            data_dir_concepts = output_dir / _class_name / "data" / "concepts"

            data_dir_maco.mkdir(parents=True, exist_ok=True)
            data_dir_concepts.mkdir(parents=True, exist_ok=True)

            pred_label = image_result["predicted_label"]
            image_filename = image_result["image_filename"]
            image_filename_clean = Path(image_filename).stem

            # Check if MACO visualization exists
            if "maco_visualization" in image_result:
                maco_image = image_result["maco_visualization"]["generated_image"]
                alpha1 = image_result["maco_visualization"]["alpha1"]
                concepts_id = image_result.get("top_k_influential_indices", [])

                # Save as npz with keywords
                maco_npy_dir = (
                    data_dir_maco / f"{image_filename_clean}" / f"{pred_label}"
                )
                maco_npy_dir.mkdir(parents=True, exist_ok=True)
                maco_npz_path = maco_npy_dir / f"maco.npz"
                np.savez_compressed(
                    maco_npz_path,
                    maco_image=maco_image.detach().cpu().numpy()
                    if isinstance(maco_image, torch.Tensor)
                    else maco_image,
                    alpha1=alpha1.detach().cpu().numpy()
                    if isinstance(alpha1, torch.Tensor)
                    else alpha1,
                    predicted_label=pred_label,
                    image_filename=image_filename,
                    concepts_id=concepts_id,
                )
                print(f"Saved MACO data to: {maco_npz_path}")
            else:
                print(
                    f"No MACO visualization found for image {i}, skipping MACO data saving"
                )

            # Save concepts as JSON
            sorted_concepts = image_result["sorted_concepts"]
            _concepts = [concept for concept, _ in sorted_concepts]
            concepts_json_dir = (
                data_dir_concepts / f"{image_filename_clean}" / f"{pred_label}"
            )
            concepts_json_dir.mkdir(parents=True, exist_ok=True)

            # Save concepts list
            concepts_json_path = concepts_json_dir / f"concepts.json"
            with open(concepts_json_path, "w") as f:
                json.dump(_concepts, f, indent=2)
            print(f"Saved concepts to: {concepts_json_path}")

            # Save sorted concepts with scores
            concepts_score_path = concepts_json_dir / f"concepts_score.json"
            with open(concepts_score_path, "w") as f:
                json.dump(sorted_concepts, f, indent=2)
            print(f"Saved concepts with scores to: {concepts_score_path}")
