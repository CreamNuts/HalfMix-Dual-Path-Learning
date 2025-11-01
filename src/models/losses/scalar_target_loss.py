from torch import nn

from ..utils import to_onehot


class ScalarTargetCE(nn.Module):
    def __init__(
        self,
        num_bins: int = 720,
        num_chunks: int = 2,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.num_chunks = num_chunks
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        """
        y_pred: shape(B, num_chunks * num_bins)
        y_true: shape(B, num_chunks)
        """
        y_preds = y_pred.chunk(self.num_chunks, dim=-1)  # shape(B, num_bins)
        y_true = to_onehot(y_true, num_bins=self.num_bins, mode="onehot")
        y_trues = y_true.chunk(self.num_chunks, dim=-1)  # shape(B,)
        losses = [self.loss(y_pred, y_true) for y_pred, y_true in zip(y_preds, y_trues)]
        return sum(losses) / len(losses)


class ScalarTargetBCE(nn.Module):
    def __init__(
        self,
        num_bins: int = 720,
        num_chunks: int = 2,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.num_chunks = num_chunks
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        """
        y_pred: shape(B, num_chunks * num_bins)
        y_true: shape(B, num_chunks)
        """
        y_preds = y_pred.chunk(self.num_chunks, dim=-1)  # shape(B, num_bins)
        y_true = to_onehot(y_true, num_bins=self.num_bins, mode="ordinal")
        y_trues = y_true.chunk(self.num_chunks, dim=-1)  # shape(B,)
        losses = [self.loss(y_pred, y_true) for y_pred, y_true in zip(y_preds, y_trues)]
        return sum(losses) / len(losses)
