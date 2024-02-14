from .augment_config import AugmentConfig
from dacapo.gp.e11 import ExpandChannels

import attr


@attr.s
class ExpandChannelsConfig(AugmentConfig):
    num_channels: int = attr.ib(
        default=0,
        metadata={"help_text": "Number of channels to expand data to"},
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, _voxel_size=None):
        return ExpandChannels(_raw_key, num_channels=self.num_channels)
