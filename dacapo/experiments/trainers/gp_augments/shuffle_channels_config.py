from .augment_config import AugmentConfig
from dacapo.gp.e11 import ShuffleChannels

import attr


@attr.s
class ShuffleChannelsConfig(AugmentConfig):
    p: float = attr.ib(
        default=1.0,
        metadata={
            "help_text": "Probability to apply the augmentation. Defaults to 1.0 (always apply)"
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        return ShuffleChannels(_raw_key, p=self.p)
