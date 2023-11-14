from .augment_config import AugmentConfig
from dacapo.gp.e11 import Blur

import attr
from typing import Tuple


@attr.s
class BlurConfig(AugmentConfig):
    blur_range: Tuple[float, float] = attr.ib(
        metadata={"help_text": "A range to randomly sample a sigma from"}
    )
    p: float = attr.ib(
        default=1.0,
        metadata={
            "help_text": "Probability to apply the augmentation. Defaults to 1.0 (always apply)"
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, _voxel_size=None):
        return Blur(
            _raw_key,
            blur_range=self.blur_range,
            p=self.p,
        )
