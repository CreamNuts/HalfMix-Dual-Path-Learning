import torch.nn.functional as F
from torch import nn


def weight_ce(weight):
    w1, w2 = weight.chunk(2, dim=0)
    # # 2 kl
    # ce = (
    #     F.cross_entropy(w1, w2.softmax(dim=1)) + F.cross_entropy(w2, w1.softmax(dim=1))
    # ) / 2
    ce = (
        F.kl_div(w1.log_softmax(dim=1), w2.softmax(dim=1), reduction="batchmean")
        + F.kl_div(w2.log_softmax(dim=1), w1.softmax(dim=1), reduction="batchmean")
    ) / 2

    return ce

def improved_weight_ce(weight):
    w1, w2 = weight.chunk(2, dim=0)
    
    # L2 정규화를 통해 각 가중치 벡터를 단위 벡터로 변환하여 크기가 아닌 방향에 집중
    w1_norm = F.normalize(w1, p=2, dim=1)
    w2_norm = F.normalize(w2, p=2, dim=1)
    
    # 정규화된 가중치에 softmax 적용
    w1_prob = F.softmax(w1_norm, dim=1)
    w2_prob = F.softmax(w2_norm, dim=1)
    
    # KL divergence 계산
    kl_div = (
        F.kl_div(w1_prob.log(), w2_prob, reduction="batchmean")
        + F.kl_div(w2_prob.log(), w1_prob, reduction="batchmean")
    ) / 2
    
    return kl_div

def weight_cs(weight):
    w1, w2 = weight.chunk(2, dim=0)
    # 1 cossim
    cs = F.cosine_similarity(w1, w2, dim=1)
    # cs = abs(cs)
    return cs.mean()


class CSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight):
        return weight_cs(weight)


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight):
        return weight_ce(weight)
