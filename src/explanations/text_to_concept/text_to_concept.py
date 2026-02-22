from typing import Any, List, Tuple, Dict, Optional
import pathlib

import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from src.explanations.text_to_concept.linear_aligner import LinearAligner
import clip
import scipy
import os
from pydantic import BaseModel, Field, ConfigDict

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORMATION = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
CLIP_IMAGENET_TRANSFORMATION = transforms.Compose(
    [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
)


class ClipZeroShot(torch.nn.Module):
    def __init__(self, mtype):
        super(ClipZeroShot, self).__init__()
        self.clip_model, self.clip_preprocess = clip.load(mtype)
        self.to_pil = transforms.ToPILImage()
        self.mtype = mtype
        self.has_normalizer = False

    def forward_features(self, img):
        image_features = self.clip_model.encode_image(img)

        return image_features

    def encode_text(self, tokens):
        return self.clip_model.encode_text(tokens)


class ZeroShotClassifier:
    def __init__(self, model, aligner: LinearAligner, zeroshot_weights: torch.Tensor):
        self.model = model
        self.aligner = aligner
        self.zeroshot_weights = zeroshot_weights.float()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # this functions returns logits.
    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            reps = self.model.forward_features(x.to(self.device)).flatten(1)
            aligned_reps = self.aligner.get_aligned_representation(reps)
            aligned_reps /= aligned_reps.norm(dim=-1, keepdim=True)

            return aligned_reps @ self.zeroshot_weights.T


class TextToConcept(BaseModel):
    """TextToConcept class using Pydantic BaseModel"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration fields
    model: Any = Field(default=None, description="Main model")
    model_name: str = Field(..., description="Model name")
    device: str = Field(default="cuda:0", description="Computing device (cpu, cuda:0)")
    clip_model_name: str = Field(default="ViT-B/16", description="CLIP model name")
    batch_size: int = Field(default=8, description="Batch size for data loader", gt=0)
    num_workers: int = Field(
        default=8, description="Number of workers for data loader", ge=0
    )
    pin_memory: bool = Field(default=True, description="Whether to use memory pinning")
    similarity_batch_size: int = Field(
        default=100, description="Batch size for similarity computation", gt=0
    )
    text_encoding_batch_size: int = Field(
        default=64, description="Batch size for text encoding", gt=0
    )

    # Model-related (set after initialization)
    clip_model: Optional[ClipZeroShot] = Field(default=None, description="CLIP model")
    linear_aligner: Optional[LinearAligner] = Field(
        default=None, description="Linear aligner"
    )

    # Data
    reps_model: Optional[np.ndarray] = Field(
        default=None, description="Model representations"
    )
    reps_clip: Optional[np.ndarray] = Field(
        default=None, description="CLIP representations"
    )
    saved_dsets: Dict[str, Tuple[str, str]] = Field(
        default_factory=dict, description="Saved datasets"
    )

    def model_post_init(self, __context) -> None:
        """Initialize after Pydantic validation"""
        # Set device automatically if not already set
        if hasattr(self, "device") and isinstance(self.device, str):
            self.device = torch.device(
                self.device
                if torch.cuda.is_available() or self.device == "cpu"
                else "cpu"
            )
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize CLIP model
        self.clip_model = ClipZeroShot(self.clip_model_name)

        # Move models to device and set to evaluation mode
        if self.model is not None:
            self.model.eval().to(self.device)
        if self.clip_model is not None:
            self.clip_model.eval().to(self.device)

    def save_reps(self, path_to_reps: pathlib.Path) -> None:
        """Save representations"""
        print(f"Saving representations")
        path_to_reps.mkdir(parents=True, exist_ok=True)
        np.save(path_to_reps / "model_reps.npy", self.reps_model)
        np.save(path_to_reps / "clip_reps.npy", self.reps_clip)

    def load_reps(self, path_to_reps: pathlib.Path) -> None:
        """Load representations"""
        print(f"Loading representations...")
        self.reps_model = np.load(path_to_reps / "model_reps.npy")
        self.reps_clip = np.load(path_to_reps / "clip_reps.npy")

    def load_linear_aligner(self, path_to_load: str) -> None:
        """Load linear aligner"""
        self.linear_aligner = LinearAligner()
        self.linear_aligner.load_W(path_to_load)

    def save_linear_aligner(self, path_to_save: str) -> None:
        """Save linear aligner"""
        if self.linear_aligner is None:
            raise ValueError("No linear aligner to save")
        self.linear_aligner.save_W(path_to_save)

    def train_linear_aligner(
        self,
        D,
        save_reps: bool = False,
        load_reps: bool = False,
        epochs: int = 5,
        path_to_reps: Optional[pathlib.Path] = None,
        writer=None,
    ) -> None:
        """Train linear aligner"""
        if load_reps:
            if path_to_reps is None:
                raise ValueError("Paths must be specified when load_reps is True")
            self.load_reps(path_to_reps)
        else:
            print(f"Obtaining representations...")
            self.reps_model = self.obtain_ftrs(self.model, D)
            self.reps_clip = self.obtain_ftrs(self.clip_model, D)

        if save_reps:
            if path_to_reps is None:
                raise ValueError("Paths must be specified when save_reps is True")
            self.save_reps(path_to_reps)

        self.linear_aligner = LinearAligner()
        self.linear_aligner.train(
            self.reps_model,
            self.reps_clip,
            epochs=epochs,
            target_variance=4.5,
            writer=writer,
        )
        del self.reps_model, self.reps_clip
        torch.cuda.empty_cache()

    def get_zeroshot_weights(
        self, classes: List[str], prompts: List[str]
    ) -> torch.Tensor:
        """Get zero-shot weights"""
        zeroshot_weights = []
        for c in classes:
            tokens = clip.tokenize([prompt.format(c) for prompt in prompts])
            c_vecs = self.clip_model.encode_text(tokens.to(self.device))
            c_vec = c_vecs.mean(0)
            c_vec /= c_vec.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(c_vec)

        return torch.stack(zeroshot_weights)

    def get_zero_shot_classifier(
        self, classes: List[str], prompts: List[str] = None
    ) -> ZeroShotClassifier:
        """Get zero-shot classifier"""
        if prompts is None:
            prompts = ["a photo of {}."]

        if self.linear_aligner is None:
            raise ValueError("Linear aligner is not trained")

        return ZeroShotClassifier(
            self.model, self.linear_aligner, self.get_zeroshot_weights(classes, prompts)
        )

    def search(
        self, dset, dset_name: str, prompts: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute concept search"""
        if prompts is None:
            prompts = ["a photo of a dog"]

        tokens = clip.tokenize(prompts)
        vecs = self.clip_model.encode_text(tokens.to(self.device))
        vec = vecs.detach().mean(0).float().unsqueeze(0)
        vec /= vec.norm(dim=-1, keepdim=True)
        sims = self.get_similarity(dset, dset_name, self.model.has_normalizer, vec)[
            :, 0
        ]
        return np.argsort(-1 * sims), sims

    def search_with_encoded_concepts(
        self, dset, dset_name: str, vec: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search with encoded concepts"""
        sims = self.get_similarity(
            dset, dset_name, self.model.has_normalizer, vec.to(self.device)
        )[:, 0]
        return np.argsort(-1 * sims), sims

    def get_similarity(
        self, dset, dset_name: str, do_normalization: bool, vecs: torch.Tensor
    ) -> np.ndarray:
        """Calculate similarity"""
        if self.linear_aligner is None:
            raise ValueError("Linear aligner is not trained")

        reps, labels = self.get_dataset_reps(dset, dset_name, do_normalization)
        N = reps.shape[0]
        batch_size = self.similarity_batch_size

        all_sims = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                aligned_reps = self.linear_aligner.get_aligned_representation(
                    torch.from_numpy(reps[i : i + batch_size]).to(self.device)
                )
                aligned_reps /= aligned_reps.norm(dim=-1, keepdim=True)
                sims = aligned_reps @ vecs.T
                sims = sims.detach().cpu().numpy()
                all_sims.append(sims)

        return np.vstack(all_sims)

    def get_dataset_reps(
        self, dset, dset_name: str, do_normalization: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get dataset representations"""
        if dset_name in self.saved_dsets:
            path_to_reps, path_to_labels = self.saved_dsets[dset_name]
            return np.load(path_to_reps), np.load(path_to_labels)

        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        all_reps, all_labels = [], []
        with torch.no_grad():
            for data in tqdm(loader):
                imgs, labels = data[0], data[1]
                if do_normalization:
                    imgs = self.model.get_normalizer(imgs).to(self.device)
                else:
                    imgs = imgs.to(self.device)

                reps = self.model.forward_features(imgs).flatten(1)
                all_reps.append(reps.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        all_reps = np.vstack(all_reps)
        all_labels = np.hstack(all_labels)

        self.saved_dsets[dset_name] = (
            self._get_path_to_reps(dset_name),
            self._get_path_to_labels(dset_name),
        )
        os.makedirs(f"datasets/{self.model_name}/", exist_ok=True)

        np.save(self._get_path_to_reps(dset_name), all_reps)
        np.save(self._get_path_to_labels(dset_name), all_labels)

        return all_reps, all_labels

    def _get_path_to_labels(self, dset_name: str) -> str:
        """Get path to labels file"""
        return f"datasets/{self.model_name}/{dset_name}_labels.npy"

    def _get_path_to_reps(self, dset_name: str) -> str:
        """Get path to representations file"""
        return f"datasets/{self.model_name}/{dset_name}_reps.npy"

    def encode_text(self, list_of_prompts: List[List[str]]) -> torch.Tensor:
        """Encode text"""
        all_vecs = []
        batch_size = self.text_encoding_batch_size

        with torch.no_grad():
            for prompts in list_of_prompts:
                tokens = clip.tokenize(prompts)
                M = tokens.shape[0]
                curr_vecs = []

                for i in range(0, M, batch_size):
                    vecs = (
                        self.clip_model.encode_text(
                            tokens[i : i + batch_size].to(self.device)
                        )
                        .detach()
                        .cpu()
                    )
                    curr_vecs.append(vecs)

                vecs = torch.vstack(curr_vecs)
                vec = vecs.mean(0).float()
                vec /= vec.norm(dim=-1, keepdim=True)
                all_vecs.append(vec)

        return torch.stack(all_vecs).to(self.device)

    def detect_drift(
        self, dset1, dset_name1: str, dset2, dset_name2: str, prompts: List[str]
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """Detect drift"""
        vecs = self.encode_text([prompts])
        sims1 = self.get_similarity(dset1, dset_name1, self.model.has_normalizer, vecs)
        sims2 = self.get_similarity(dset2, dset_name2, self.model.has_normalizer, vecs)

        stats, p_value = scipy.stats.ttest_ind(sims1[:, 0], sims2[:, 0])
        return [stats, p_value], sims1, sims2

    def concept_logic(
        self,
        dset,
        dset_name: str,
        list_of_prompts: List[List[str]],
        signs: List[int],
        scales: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute concept logic"""
        vecs = self.encode_text(list_of_prompts)
        sims = self.get_similarity(dset, dset_name, self.model.has_normalizer, vecs)
        means = np.mean(sims, axis=0)
        stds = np.std(sims, axis=0)

        ths = means + np.array(signs) * np.array(scales) * stds
        retrieved = np.arange(sims.shape[0])

        for j in range(len(signs)):
            if retrieved.shape[0] == 0:
                break

            sim_to_concept = sims[retrieved, j]
            if signs[j] == -1:
                retrieved = retrieved[np.where(sim_to_concept < ths[j])[0]]
            else:
                retrieved = retrieved[np.where(sim_to_concept > ths[j])[0]]

        return retrieved, sims

    def obtain_ftrs(self, model, dset) -> np.ndarray:
        """Obtain features"""
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=16,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return self.obtain_reps_given_loader(model, loader)

    def obtain_reps_given_loader(self, model, loader) -> np.ndarray:
        """Obtain representations from loader"""
        all_reps = []
        for imgs, _ in tqdm(loader):
            if model.has_normalizer:
                imgs = model.get_normalizer(imgs)

            imgs = imgs.to(self.device)
            reps = model.forward_features(imgs).flatten(1)
            reps = [x.detach().cpu().numpy() for x in reps]
            all_reps.extend(reps)

        all_reps = np.stack(all_reps)
        return all_reps

    def select_best_caption(
        self, image: torch.Tensor, caption_candidates: List[str]
    ) -> Tuple[str, np.ndarray, int]:
        """
        Select the caption with highest similarity to the image from multiple candidates

        Args:
            image: Input image (tensor)
            caption_candidates: List of caption candidates

        Returns:
            best_caption: Caption with highest similarity
            similarities: Similarity scores for each caption
            best_index: Index of the caption with highest similarity
        """
        if self.linear_aligner is None:
            raise ValueError("Linear aligner is not trained")

        with torch.no_grad():
            # Get image features
            if self.model.has_normalizer:
                image = self.model.get_normalizer(image.unsqueeze(0))
            image_features = self.model.forward_features(image.to(self.device)).flatten(
                1
            )
            aligned_image_features = self.linear_aligner.get_aligned_representation(
                image_features
            )
            aligned_image_features /= aligned_image_features.norm(dim=-1, keepdim=True)

            # Encode caption candidates
            text_features = []
            for caption in caption_candidates:
                tokens = clip.tokenize([caption]).to(self.device)
                text_vec = self.clip_model.encode_text(tokens)
                text_vec /= text_vec.norm(dim=-1, keepdim=True)
                text_features.append(text_vec)

            text_features = torch.cat(text_features, dim=0)

            # Calculate similarity
            similarities = (aligned_image_features @ text_features.T).squeeze(0)
            similarities = similarities.detach().cpu().numpy()

            # Select caption with highest similarity
            best_index = np.argmax(similarities)
            best_caption = caption_candidates[best_index]

            return best_caption, similarities, best_index

    def rank_captions_by_similarity(
        self, image: torch.Tensor, caption_candidates: List[str]
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Rank multiple caption candidates by similarity to the image

        Args:
            image: Input image (tensor)
            caption_candidates: List of caption candidates

        Returns:
            ranked_captions: List of captions sorted by similarity
            ranked_similarities: Corresponding similarity scores
            ranked_indices: Original indices
        """
        best_caption, similarities, best_index = self.select_best_caption(
            image, caption_candidates
        )

        # Sort by similarity
        ranked_indices = np.argsort(-similarities)  # Descending order
        ranked_captions = [caption_candidates[i] for i in ranked_indices]
        ranked_similarities = similarities[ranked_indices]

        return ranked_captions, ranked_similarities, ranked_indices
