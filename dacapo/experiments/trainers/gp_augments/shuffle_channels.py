from .augment_config import AugmentConfig
from dacapo.gp.e11 import ShuffleChannels

import attr


@attr.s
class ShuffleChannelsConfig(AugmentConfig):
    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        return ShuffleChannels(_raw_key)
