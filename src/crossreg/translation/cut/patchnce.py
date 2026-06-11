"""
PatchNCE loss for CUT contrastive learning.

From: https://github.com/taesungp/contrastive-unpaired-translation
Paper: Contrastive Learning for Unpaired Image-to-Image Translation (ECCV 2020)
"""

from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    """Patch-wise contrastive loss used in CUT.

    Args:
        nce_T: temperature for NCE loss.
        batch_size: batch size (used when nce_includes_all_negatives_from_minibatch=True).
        nce_includes_all_negatives_from_minibatch:
            If True, negative samples are drawn from the entire minibatch.
            Set False for standard CUT, True for single-image translation.
    """

    def __init__(
        self,
        nce_T: float = 0.07,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
    ):
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = (
            torch.uint8 if version.parse(torch.__version__) < version.parse("1.2.0") else torch.bool
        )

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        """Compute PatchNCE loss.

        Args:
            feat_q: query features, shape (num_patches, dim)
            feat_k: key features, shape (num_patches, dim)

        Returns:
            scalar loss (mean over patches).
        """
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # positive logit
        l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # negative logit
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # fill diagonal with very small number (exp(-10) ≈ 0)
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss
