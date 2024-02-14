from .augment_config import AugmentConfig
from dacapo.gp.contrib import ChannelWiseNoiseAugment

import attr


@attr.s
class ChannelWiseNoiseAugmentConfig(AugmentConfig):
    p: float = attr.ib(
        default=1.0,
        metadata={
            "help_text": "Probability to apply the augmentation. Defaults to 1.0 (always apply)"
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, _voxel_size=None):
        return ChannelWiseNoiseAugment(
            _raw_key,
            p=self.p,
        )
