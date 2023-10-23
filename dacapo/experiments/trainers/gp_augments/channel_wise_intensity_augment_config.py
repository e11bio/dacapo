from .augment_config import AugmentConfig
from dacapo.gp.e11 import ChannelWiseIntensityAugment

import attr
from typing import Tuple


@attr.s
class ChannelWiseIntensityAugmentConfig(AugmentConfig):
    scale: Tuple[float, float] = attr.ib(
        metadata={"help_text": "A range within which to choose a random scale factor."}
    )
    shift: Tuple[float, float] = attr.ib(
        metadata={
            "help_text": "A range within which to choose a random additive shift."
        }
    )
    clip: bool = attr.ib(
        default=True,
        metadata={
            "help_text": "Set to False if modified values should not be clipped to [0, 1]"
        },
    )
    z_section_wise: bool = attr.ib(
        default=False,
        metadata={
            "help_text": (
                "Perform the augmentation z-section wise. Assumes z is"
                "the first spatial dimension."
            )
        },
    )
    p: float = attr.ib(
        default=1.0,
        metadata={
            "help_text": "Probability to apply the augmentation. Defaults to 1.0 (always apply)"
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        return ChannelWiseIntensityAugment(
            _raw_key,
            scale_min=self.scale[0],
            scale_max=self.scale[1],
            shift_min=self.shift[0],
            shift_max=self.shift[1],
            clip=self.clip,
            z_section_wise=self.z_section_wise,
            p=self.p,
        )
