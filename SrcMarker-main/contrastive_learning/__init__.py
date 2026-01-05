"""Contrastive learning package for robust code representation"""

from .model import RobustEncoder
from .augmentor import CodeAugmentor
from .dataset import ContrastiveTrainDataset, CloneDetectionTestDataset
from .losses import InfoNCELoss
from .trainer import ContrastiveTrainer

__all__ = [
    'RobustEncoder',
    'CodeAugmentor',
    'ContrastiveTrainDataset',
    'CloneDetectionTestDataset',
    'InfoNCELoss',
    'ContrastiveTrainer'
] 