import torch
import torch.nn.functional as F
from torch import nn

from ..utils import to_onehot


class SubjectLoss(nn.Module):
    def __init__(self, temperature=1, eps=1e-6) -> None:
        super(SubjectLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, subjects):
        """
        features: shape(B, d)
        subjects: shape(B,)
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        bsz = features.shape[0]

        # compute label similarity matrix
        subjects = subjects.unsqueeze(1).float()
        subject_dist = torch.cdist(subjects, subjects, p=1)
        subject_dist = torch.where(subject_dist >= 1, 2.0, 0.0)
        subject_similarity = subject_dist - 1
        # compute feature similarity matrix
        l2_norm = F.normalize(features, dim=1, p=2)
        feature_similarity = torch.exp(torch.mm(l2_norm, l2_norm.T) / self.temperature)

        # compute mask
        mask_out = 1 - torch.eye(bsz, device=device).float()

        positive = torch.sum(
            F.relu(mask_out * subject_similarity) * feature_similarity, dim=1
        )
        negative = (
            torch.sum(
                torch.abs(mask_out * subject_similarity) * feature_similarity, dim=1
            )
            + self.eps
        )
        loss = torch.log(positive / negative + self.eps)
        loss = -torch.mean(loss)
        return loss
