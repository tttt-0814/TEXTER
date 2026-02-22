"""Sparse autoencoder modules and training helpers."""

from .base import SAE
from .archetypal_dictionary import RelaxedArchetypalDictionary
from .batchtopk_sae import BatchTopKSAE
from .dictionary import DictionaryLayer
from .factory import EncoderFactory
from .jump_sae import JumpSAE, heaviside, jump_relu
from .losses import mse_l1
from .modules import AttentionEncoder, MLPEncoder, ResNetEncoder
from .mp_sae import MpSAE
from .omp_sae import OMPSAE
from .optimizer import CosineScheduler
from .qsae import QSAE
from .topk_sae import TopKSAE
from .train import trainer_sae

__all__ = [
    "AttentionEncoder",
    "BatchTopKSAE",
    "CosineScheduler",
    "DictionaryLayer",
    "EncoderFactory",
    "JumpSAE",
    "MLPEncoder",
    "MpSAE",
    "OMPSAE",
    "QSAE",
    "RelaxedArchetypalDictionary",
    "ResNetEncoder",
    "SAE",
    "TopKSAE",
    "heaviside",
    "jump_relu",
    "mse_l1",
    "trainer_sae",
]
