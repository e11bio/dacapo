from .augment_config import AugmentConfig

import gunpowder as gp

import attr


@attr.s
class SimpleAugmentConfig(AugmentConfig):
    def node(
        self,
        _raw_key=None,
        _gt_key=None,
        _mask_key=None,
        mirror_only=None,
        transpose_only=None,
        mirror_probs=None,
        transpose_probs=None,
    ):
        return gp.SimpleAugment(
            mirror_only, transpose_only, mirror_probs, transpose_probs
        )
