import torch
import torch.nn as nn


def get_loss_mask_77(device: str = "cpu") -> torch.Tensor:
    mask = torch.ones(77, device=device)
    mask[66] = 0
    mask[67] = 0
    mask[68] = 0
    return mask


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask: torch.Tensor | None = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.mask is None or self.mask.device != pred.device:
            self.mask = get_loss_mask_77(pred.device)

        diff = (pred - target) ** 2
        masked_diff = diff * self.mask.unsqueeze(0)
        loss = masked_diff.sum() / (self.mask.sum() * pred.shape[0])
        return loss
