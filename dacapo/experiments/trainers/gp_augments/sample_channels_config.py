from .augment_config import AugmentConfig
from dacapo.gp.contrib import SampleChannels

import attr


@attr.s
class SampleChannelsConfig(AugmentConfig):
    num_channels: int = attr.ib(
        metadata={"help_text": "The number of channels to take."}
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, _voxel_size=None):
        return SampleChannels(_raw_key, self.num_channels)
