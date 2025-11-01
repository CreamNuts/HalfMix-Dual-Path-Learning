from torch import nn
from ..utils import to_onehot


class OneHotInputCE(nn.Module):
    def __init__(
        self,
        num_bins: int = 720,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        """
        y_pred: shape(B, 2 * num_bins)
        y_true: shape(B, 2 * num_bins) or shape(B, 2)
        """
        if y_true.size(-1) == 2:
            y_true = to_onehot(y_true, num_bins=self.num_bins, mode="onehot")
        y_pred1, y_pred2 = y_pred.chunk(2, dim=-1)  # shape(B, num_bins)
        y_true1, y_true2 = y_true.chunk(2, dim=-1)
        loss1 = self.loss(y_pred1, y_true1)
        loss2 = self.loss(y_pred2, y_true2)
        loss = (loss1 + loss2) / 2
        return loss
