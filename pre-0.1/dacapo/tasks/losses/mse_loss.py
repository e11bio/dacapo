from .loss_abc import LossABC

import torch
import attr


@attr.s
class MSELoss(LossABC):

    def instantiate(self):
        return torch.nn.MSELoss()