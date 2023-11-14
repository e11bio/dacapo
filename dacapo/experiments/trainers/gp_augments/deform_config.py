from .augment_config import AugmentConfig
from gunpowder.nodes import DeformAugment
from funlib.geometry import Coordinate

import attr

from typing import List, Tuple


@attr.s
class DeformAugmentConfig(AugmentConfig):
    control_point_spacing: List[int] = attr.ib(
        metadata={
            "help_text": (
                "Distance between control points for the elastic deformation, in "
                "voxels per dimension."
            )
        }
    )
    control_point_displacement_sigma: List[float] = attr.ib(
        metadata={
            "help_text": (
                "Standard deviation of control point displacement distribution, in world coordinates."
            )
        }
    )
    scale_interval: Tuple[float, float] = attr.ib(
        default=(0.9, 1.1),
        metadata={"help_text": ("Interval to randomly sample scaling factors from")},
    )
    subsample: int = attr.ib(
        default=1,
        metadata={
            "help_text": "Perform the elastic augmentation on a grid downsampled by this factor."
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, voxel_size=None):
        return DeformAugment(
            control_point_spacing=Coordinate(self.control_point_spacing)
            * voxel_size[1],
            jitter_sigma=Coordinate(self.control_point_displacement_sigma)
            * voxel_size[1],
            scale_interval=self.scale_interval,
            subsample=self.subsample,
        )
