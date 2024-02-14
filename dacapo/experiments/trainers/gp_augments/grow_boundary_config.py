from .augment_config import AugmentConfig
from dacapo.gp.e11 import CustomGrowBoundary
import attr
import gunpowder as gp


@attr.s
class GrowBoundaryConfig(AugmentConfig):
    steps: int = attr.ib(
        default=1,
        metadata={"help_text": ("Number of voxels (not world units!) to grow.")},
    )
    background: int = attr.ib(
        default=0,
        metadata={"help_text": ("The label to assign to the boundary voxels.")},
    )
    only_xy: bool = attr.ib(
        default=False,
        metadata={"help_text": "Do not grow a boundary in the z direction."},
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, _voxel_size=None):
        return CustomGrowBoundary(
            labels=_gt_key,
            mask=_mask_key,
            steps=self.steps,
            background=self.background,
            only_xy=self.only_xy,
        )
