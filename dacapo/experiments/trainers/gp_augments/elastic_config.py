from .augment_config import AugmentConfig
from typing import List, Tuple
import attr
import gunpowder as gp


@attr.s
class ElasticAugmentConfig(AugmentConfig):
    control_point_spacing: List[int] = attr.ib(
        metadata={
            "help_text": (
                "Distance between control points for the elastic deformation, in "
                "voxels per dimension."
            )
        }
    )
    jitter_sigma: List[float] = attr.ib(
        metadata={
            "help_text": (
                "Standard deviation of control point displacement distribution, in world coordinates."
            )
        }
    )
    rotation_interval: Tuple[float, float] = attr.ib(
        metadata={
            "help_text": ("Interval to randomly sample rotation angles from (0, 2PI).")
        }
    )
    scale_interval: Tuple[float, float] = attr.ib(
        default=(1.0, 1.0),
        metadata={"help_text": ("Interval to randomly scale factors from")}
    )
    subsample: int = attr.ib(
        default=1,
        metadata={
            "help_text": "Perform the elastic augmentation on a grid downsampled by this factor."
        },
    )
    p: float = attr.ib(
        default=1.0,
        metadata={
            "help_text": "Probability to apply the augmentation. Defaults to 1.0 \
                (always apply)"
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, _voxel_size=None):
        return gp.ElasticAugment(
            control_point_spacing=self.control_point_spacing,
            jitter_sigma=self.jitter_sigma,
            rotation_interval=self.rotation_interval,
            scale_interval=self.scale_interval,
            subsample=self.subsample,
            p=self.p,
        )
