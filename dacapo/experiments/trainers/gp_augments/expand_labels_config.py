from .augment_config import AugmentConfig
from dacapo.gp.e11 import ExpandLabels

import attr


@attr.s
class ExpandLabelsConfig(AugmentConfig):
    background: int = attr.ib(
        default=0, metadata={"help_text": ("The label belonging to background.")}
    )
    expansion_factor: int = attr.ib(
        default=1, metadata={"help_text": ("The number of voxels to expand into.")}
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        return ExpandLabels(_gt_key, self.background, self.expansion_factor)
