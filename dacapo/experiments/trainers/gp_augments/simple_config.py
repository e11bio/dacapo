from .augment_config import AugmentConfig
from typing import Dict, Tuple, Union, List, Optional

import gunpowder as gp

import attr


@attr.s
class SimpleAugmentConfig(AugmentConfig):
    mirror_only: Optional[List[int]] = attr.ib(
        default=None,
        metadata={
            "help_text": (
                "If set, only mirror between the given axes. This is useful to exclude"
                "channels that have a set direction, like time."
            )
        },
    )
    transpose_only: Optional[List[int]] = attr.ib(
        default=None,
        metadata={
            "help_text": (
                "If set, only transpose between the given axes. This is useful to limit"
                "the transpose to axes with the same resolution or to exclude"
                "non-spatial dimensions."
            )
        },
    )
    mirror_probs: Optional[List[float]] = attr.ib(
        default=None,
        metadata={
            "help_text": (
                "If set, provides the probability for mirroring given axes. Default is"
                "0.5 per axis. If given, must be given for every axis. i.e. [0,1,0]"
                "for 100% chance of mirroring axis 1 an no others."
            )
        },
    )
    transpose_probs: Optional[Dict[Tuple, Union[float, List[float]]]] = attr.ib(
        default=None,
        metadata={
            "help_text": (
                "The probability of transposing. If None, each transpose is equally"
                "likely. Can also be a dictionary of for tuple -> float."
                "See gunpowder docs for examples."
            )
        },
    )
    p: float = attr.ib(
        default=1.0,
        metadata={
            "help_text": "Probability to apply the augmentation. Defaults to 1.0 \
                (always apply)"
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, _voxel_size=None):
        return gp.SimpleAugment(
            mirror_only=self.mirror_only,
            transpose_only=self.transpose_only,
            mirror_probs=self.mirror_probs,
            transpose_probs=self.transpose_probs,
            p=self.p,
        )
