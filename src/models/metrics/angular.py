import math

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from ..utils.convert import GazeTo3d, to_scalar


def compute_angular_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute angular error between two vectors.

    Args:
        preds (torch.Tensor): Predicted vectors.
        targets (torch.Tensor): Target vectors.

    Returns:
        torch.Tensor: Angular error.
    """
    with torch.no_grad():
        similarity = F.cosine_similarity(preds, targets)
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)
        angular_error = torch.arccos(similarity) * 180 / math.pi
    return angular_error


class AngularError(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        lb_type: str = "scalar",
        num_chunks: int = 2,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        assert lb_type in ["scalar", "onehot", "ordinal"]
        self.lb_type = lb_type
        self.num_chunks = num_chunks
        self.add_state("preds", [], dist_reduce_fx="cat")
        self.add_state("targets", [], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        r"""Update metric states with predictions and targets.
        Args:
            preds: Predicted tensor with shape ``(N,2)`` or ``(N,2*num_bins)``.
            target: Ground truth tensor with shape ``(N,2)``
        """
        if preds.size(1) > 3 and self.lb_type != "scalar": # (N, M*num_bins) -> (N, M), M = 2 or 3
            preds = to_scalar(preds, mode=self.lb_type, num_chunks=self.num_chunks)
        if self.num_chunks == 2:
            preds, targets = GazeTo3d(preds), GazeTo3d(targets)
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)
        angular_error = compute_angular_error(preds, targets)
        return torch.mean(angular_error)
