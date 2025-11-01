import torch
from torch import nn


class DoubleHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.head = nn.Linear(in_features, out_features * 2)
        # self.init_weights()

    def forward(self, x):
        x = self.head(x)
        out1, out2 = x.chunk(2, dim=-1)
        return out1, out2

    def init_weights(self):
        # 가중치의 전체 크기를 출력 특성 수의 두 배로 설정하고 있기 때문에
        # 가중치를 두 부분으로 나누어 처리합니다.
        with torch.no_grad():
            # 가중치 행렬의 크기는 (out_features * 2, in_features)
            weight = self.head.weight
            # 가중치 행렬을 두 부분으로 분할
            w1, w2 = weight.chunk(2, dim=0)
            # 첫 번째 절반(w1)을 랜덤하게 초기화
            nn.init.xavier_uniform_(w1)
            # 두 번째 절반(w2)을 첫 번째의 정확히 반대 방향으로 설정
            w2.copy_(-w1)

            bias = self.head.bias
            b1, b2 = bias.chunk(2, dim=0)
            b2.copy_(-b1)


class DoubleEmbeddingHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj1 = nn.Linear(in_features, in_features)
        self.proj2 = nn.Linear(in_features, in_features)
        # self.head = nn.Linear(in_features, out_features)
        self.head1 = nn.Linear(in_features, out_features)
        self.head2 = nn.Linear(in_features, out_features)

    def forward_proj(self, x):
        return self.proj1(x), self.proj2(x)

    def forward(self, x):
        f1, f2 = self.forward_proj(x)
        out1 = self.head1(f1)
        out2 = self.head2(f2)
        # out1 = self.head(f1)
        # out2 = self.head(f2)
        if self.training:
            return out1, out2, f1, f2, x
        else:
            return out1, out2


class HalfDoubleHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert in_features % 2 == 0, "in_features must be divisible by 2"
        self.head1 = nn.Linear(in_features // 2, out_features)
        self.head2 = nn.Linear(in_features // 2, out_features)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.head1(x1), self.head2(x2)


class MLPHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, out_features),
        )

    def forward(self, x):
        return self.head(x)
