from .augment_config import AugmentConfig
from dacapo.gp.e11 import ZeroChannels

import attr


@attr.s
class ZeroChannelsConfig(AugmentConfig):
    num_channels: int = attr.ib(
        default=0,
        metadata={"help_text": "Number of channels to zero"},
    )
    p: float = attr.ib(
        default=1.0,
        metadata={
            "help_text": "Probability to apply the augmentation. Defaults to 1.0 (always apply)"
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        return ZeroChannels(
            _raw_key,
            num_channels=self.num_channels,
            p=self.p,
        )
