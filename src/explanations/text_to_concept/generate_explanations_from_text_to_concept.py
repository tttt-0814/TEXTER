from typing import Dict, Any, List, Union

import torch

from src.explanations.mymethod.textual_explanation import MyMethod


class SimpleTextToConcept(MyMethod):
    """Compute text-to-concept similarity with minimal steps."""

    def __call__(
        self,
        images: torch.Tensor,
        concepts: Union[List[str], List[List[str]]],
    ) -> Dict[str, Any]:
        with torch.no_grad():
            features = self.get_classification_model_features(images)
            aligned_features = self.get_aligned_features(features, normalize=True)
            concepts_features = self.get_concepts_features(concepts, normalize=True)
            similarity_scores = self.compute_similarity_scores(
                aligned_features, concepts_features, normalize_features=False
            )
            sorted_concepts = self.sort_concepts_by_similarity(
                concepts, similarity_scores
            )

        return {
            "similarity_scores": similarity_scores,
            "sorted_concepts": sorted_concepts,
        }

    def sort_concepts_by_similarity(
        self,
        concepts: Union[List[str], List[List[str]]],
        similarity_scores: Union[torch.Tensor, List[torch.Tensor]],
    ) -> List[List[tuple[str, float]]]:
        """Return concepts sorted by similarity scores (descending)."""

        concepts_per_image = self._normalize_concepts(concepts)
        if not concepts_per_image:
            return []

        if len(concepts_per_image) == 1 and (
            isinstance(similarity_scores, torch.Tensor)
            and similarity_scores.shape[0] > 1
            or isinstance(similarity_scores, list)
            and len(similarity_scores) > 1
        ):
            concepts_per_image = [
                concepts_per_image[0]
                for _ in range(
                    similarity_scores.shape[0]
                    if isinstance(similarity_scores, torch.Tensor)
                    else len(similarity_scores)
                )
            ]

        results: List[List[tuple[str, float]]] = []
        if isinstance(similarity_scores, torch.Tensor):
            for b, concept_list in enumerate(concepts_per_image):
                if not concept_list:
                    results.append([])
                    continue
                scores_b = similarity_scores[b]
                k = min(scores_b.numel(), len(concept_list))
                vals, idxs = torch.topk(scores_b, k=k)
                results.append(
                    [
                        (concept_list[i], float(vals[j].item()))
                        for j, i in enumerate(idxs.tolist())
                    ]
                )
            return results

        if not isinstance(similarity_scores, list):
            raise TypeError("similarity_scores must be a Tensor or a list of Tensors.")

        for b, scores_b in enumerate(similarity_scores):
            concept_list = concepts_per_image[b] if b < len(concepts_per_image) else []
            if not concept_list or scores_b.numel() == 0:
                results.append([])
                continue
            k = min(scores_b.numel(), len(concept_list))
            vals, idxs = torch.topk(scores_b, k=k)
            results.append(
                [
                    (concept_list[i], float(vals[j].item()))
                    for j, i in enumerate(idxs.tolist())
                ]
            )

        return results
