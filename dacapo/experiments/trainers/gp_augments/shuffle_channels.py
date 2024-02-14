from .augment_config import AugmentConfig
from dacapo.gp.contrib import ShuffleChannels

import attr


@attr.s
class ShuffleChannelsConfig(AugmentConfig):
    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, _voxel_size=None):
        return ShuffleChannels(_raw_key)
