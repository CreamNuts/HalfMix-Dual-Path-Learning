import math

import torch
import torch.nn.functional as F


def single_GazeTo3d(gaze):
    r"""Convert given pitch and yaw angles to unit gaze vectors.
    Args:
        gaze (:obj:`torch.tensor` of shape `(2)`): Pitch and yaw angles in radians.
    Returns:
        :obj:`torch.tensor` of shape `(3)`: 3D gaze vectors.
    """
    sin = torch.sin(gaze)
    cos = torch.cos(gaze)
    out = torch.empty((3)).to(gaze.device)
    out[0] = torch.multiply(cos[0], sin[1])
    out[1] = sin[0]
    out[2] = torch.multiply(cos[0], cos[1])
    return out

def GazeTo3d(gaze):
    r"""Convert given pitch and yaw angles to unit gaze vectors.
    Args:
        gaze (:obj:`torch.tensor` of shape `(N, 2)`): Pitch and yaw angles in radians.
    Returns:
        :obj:`torch.tensor` of shape `(N, 3)`: 3D gaze vectors.
    """
    n = gaze.shape[0]
    sin = torch.sin(gaze)
    cos = torch.cos(gaze)
    out = torch.empty((n, 3)).to(gaze.device)
    out[:, 0] = torch.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = torch.multiply(cos[:, 0], cos[:, 1])
    return out

# TODO: x,y,z 인 경우 min, max가 적절히 변경되어야함
@torch.no_grad()
def to_onehot(
    value: torch.Tensor,
    min: float = -math.pi,
    max: float = math.pi,
    num_bins: int = 720,
    mode: str = "onehot",
):
    """
    Args:
        value (:obj:`torch.tensor` of shape `(N, 2)` or `(N, 3)`): Pitch and yaw angles in radians. If `(N, 3)`, x, y, z components of gaze vector.
        min (float): Minimum value of the interval.
        max (float): Maximum value of the interval.
        num_bins (int): Number of bins.
        mode (str): One of "onehot" or "ordinal".
    """
    value = value.clamp(min, max)
    boundaries = torch.linspace(min, max, num_bins + 1).to(value.device)
    indices = torch.bucketize(value, boundaries, right=True) - 1
    results = []
    for i in range(value.shape[-1]):
        if mode == "onehot":
            vector = torch.zeros(
                value.shape[0], num_bins, dtype=torch.float32, device=value.device
            )
            vector[torch.arange(value.shape[0])] = F.one_hot(
                indices[torch.arange(value.shape[0]), i], num_bins
            ).float()

        elif mode == "ordinal":
            vector = torch.ones(
                value.shape[0], num_bins, dtype=torch.float32, device=value.device
            )
            for j in range(value.shape[0]):
                vector[j, indices[j, i] :] = 0

        else:
            raise ValueError("Invalid mode. Choose 'onehot' or 'ordinal'.")
        results.append(vector)
    return torch.cat(results, dim=-1)


@torch.no_grad()
def to_scalar(
    vectors: torch.Tensor,
    min: float = -math.pi,
    max: float = math.pi,
    num_chunks: int = 2,
    mode: str = "onehot",
):
    """
    Args:
        vectors (:obj:`torch.tensor` of shape `(N, M)`): Pitch and yaw angles expressed as one-hot/ordinal vectors.
        min (float): Minimum value of the interval.
        max (float): Maximum value of the interval.
        num_chunks (int): Number of chunks. M must be divisible by num_chunks.
        mode (str): One of "onehot" or "ordinal".
    """
    num_interval = (
        vectors.shape[-1] // num_chunks
    )  # vectors.shape[-1] = num_chunks * num_interval
    interval = (max - min) / num_interval

    results = []
    for i in range(num_chunks):
        vec = vectors[
            :, i * num_interval : (i + 1) * num_interval
        ]  # shape(B * 2, num_interval)

        if mode == "onehot":
            idx = torch.argmax(vec, dim=-1)

        elif mode == "ordinal":
            idx = (vec.sigmoid() > 0.5).cumprod(dim=-1).sum(dim=-1)
        else:
            raise ValueError("Invalid mode. Choose 'onehot' or 'ordinal'.")
        scalar = min + (idx + 0.5) * interval
        results.append(scalar)
    return torch.stack(results, dim=-1)
