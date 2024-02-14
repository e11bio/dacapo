from .augment_config import AugmentConfig
from dacapo.gp.e11 import UpdateMask
import attr
import gunpowder as gp


@attr.s
class UpdateMaskConfig(AugmentConfig):
    mask_type: str = attr.ib(
        default="labels_mask",
        metadata={
            "help_text": (
                "Type of mask to update, should be one of labels_mask, unlabelled"
            )
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None, _voxel_size=None):
        return UpdateMask(labels=_gt_key, mask=_mask_key, mask_type=self.mask_type)
