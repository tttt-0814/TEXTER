from typing import List, Optional, Literal, Tuple
from pydantic import Field


import torch
import clip

from src.explanations.text_to_concept import TextToConcept
from src.explanations.generator.concept_extractor import ConceptExtractor


class LLMConceptGenerator(TextToConcept):
    # Additional configuration for explanation
    top_k_concepts: int = Field(
        default=5, description="Number of top concepts to use for explanation"
    )
    confidence_threshold: float = Field(
        default=0.1, description="Minimum confidence threshold for concepts"
    )

    # Candidate generation mode
    candidate_generation_mode: Literal["llm", "vlm"] = Field(
        default="llm",
        description="Mode for generating candidates: 'llm' or 'vlm'",
    )

    # GPT configuration
    openai_api_key: Optional[str] = Field(
        None,
        description="OpenAI API key",
    )
    llm_model: str = Field(default="gpt-3.5-turbo", description="GPT model to use")
    vlm_model: str = Field(
        # default="gpt-4o",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Vision-Language model to use",
    )
    max_candidate_descriptions: int = Field(
        default=100, description="Maximum number of candidate descriptions to generate"
    )

    concept_extractor: Optional[ConceptExtractor] = Field(
        None, description="Concept extractor"
    )
    image_features: Optional[torch.Tensor] = Field(
        default=None, description="Current image features"
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._initialize_concept_extractor()

    def _initialize_concept_extractor(self) -> None:
        """Initialize ConceptExtractor"""
        self.concept_extractor = ConceptExtractor(
            api_key=self.openai_api_key,
            llm_model=self.llm_model,
            vlm_model=self.vlm_model,
            max_total_descriptions=self.max_candidate_descriptions,
        )

    def compute_concept_similarity_for_maco_features(
        self,
        maco_features: torch.Tensor,
        candidate_concepts: List[str],
    ) -> Tuple[str, List[Tuple[str, float]], torch.Tensor, int]:
        self.image_features = maco_features
        # Encode descriptions and image
        concept_embeddings = self.encode_concepts(candidate_concepts)

        # Compute similarities for each maco feature
        all_similarities = []
        for i in range(maco_features.shape[0]):
            maco_feature = maco_features[i]
            # Select best description
            _, similarities, _ = self.compute_concepts_similarity(
                candidate_concepts, concept_embeddings, maco_feature[None, ...]
            )
            all_similarities.append(similarities)

        # Average similarities across all maco features
        averaged_similarities = torch.stack(all_similarities).mean(dim=0)

        # Sort concepts by averaged similarity
        sorted_indices = torch.argsort(averaged_similarities, descending=True)
        sorted_concepts_with_similarity = [
            (candidate_concepts[idx], averaged_similarities[idx].item())
            for idx in sorted_indices
        ]

        return sorted_concepts_with_similarity, averaged_similarities, sorted_indices

    def encode_concepts(
        self, concepts: List[str], normalize: bool = True
    ) -> torch.Tensor:
        """Encode descriptions using CLIP text encoder"""
        if not concepts:
            return torch.empty(0, self.clip_model.text_projection.out_features)

        text_tokens = clip.tokenize(concepts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Ensure float32 dtype for consistency
            text_features = text_features.float()
        return text_features

    def get_img_feature_from_aligned_reps(self, imgs, normalize: bool = True):
        if self.model.has_normalizer:
            imgs = self.model.get_normalizer(imgs).to(self.device)
        else:
            imgs = imgs.to(self.device)

        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            img_features = self.model.forward_features(imgs)
            img_features = img_features.flatten(1)
            aligned_image_features = self.linear_aligner.get_aligned_representation(
                img_features
            )
            if normalize:
                aligned_image_features = (
                    aligned_image_features
                    / aligned_image_features.norm(dim=-1, keepdim=True)
                )
            # Ensure float16 dtype for consistency
            aligned_image_features = aligned_image_features.float()

        return aligned_image_features.detach()

    def compute_concepts_similarity(
        self,
        concepts: List[str],
        concept_embeddings: torch.Tensor,
        img_embedding: torch.Tensor,
    ) -> Tuple[List[Tuple[str, float]], torch.Tensor, List[int]]:
        # Ensure both tensors have the same dtype (float32)
        concept_embeddings = concept_embeddings.float()
        img_embedding = img_embedding.float()

        # Calculate similarities
        similarities = torch.matmul(img_embedding, concept_embeddings.T).squeeze()

        if similarities.dim() == 0:  # Single description case
            similarities = similarities.unsqueeze(0)

        # Sort by similarity (descending order)
        sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)

        # Create list of (concept, similarity) tuples sorted by similarity
        sorted_concepts_with_similarity = [
            (concepts[idx.item()], sorted_similarities[i].item())
            for i, idx in enumerate(sorted_indices)
        ]

        return sorted_concepts_with_similarity, similarities, sorted_indices.tolist()
