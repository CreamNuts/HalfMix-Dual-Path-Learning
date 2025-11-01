from typing import Iterable, Optional

from torch.optim import Optimizer, RAdam


def add_weight_decay(model, weight_decay=1e-6, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def create_optimizer(
    model,
    opt: Optimizer = RAdam,
    lr: Optional[float] = None,
    weight_decay: float = 0.0,
    no_weight_decay_list: Iterable = (),
):
    if weight_decay:
        parameters = add_weight_decay(
            model=model,
            weight_decay=weight_decay,
            no_weight_decay_list=no_weight_decay_list,
        )
    else:
        parameters = model.parameters()

    optimizer = opt(parameters, lr=lr)
    return optimizer
