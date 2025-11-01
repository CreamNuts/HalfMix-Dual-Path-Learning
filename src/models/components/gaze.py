import torch
from timm import create_model
from torch import nn


class GazeModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        num_chunks: int = 2,
        head_func: nn.Module = nn.Linear,
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.head_func = head_func
        self.backbone = create_model(model_name, pretrained=pretrained, num_classes=0)
        self.set_head(self.num_chunks, self.head_func)

    @torch.no_grad()
    def set_head(self, num_chunks, head_func=None):
        self.eval()
        if head_func is None:
            head_func = self.head_func
        test = torch.randn(1, 3, 224, 224)
        num_features = self.backbone(test).shape[-1]
        self.head = head_func(num_features, num_chunks)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.forward_features(x)
        gaze = self.head(features)
        return gaze
    
    @property
    def features(self):
        """Property to access features method for SAE integration."""
        return self.forward_features
