from typing import List, Sequence
from torch import Tensor
import torch


@torch.inference_mode()
def topk_accuracy(output: Tensor, target: Tensor, topk: List[int] = (1,)) -> List[Tensor]:
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L411

    Args:
        output: [B, C], for C way classification
        target: [B]
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if target.ndim == 2:
        # Possibly onehot target
        target = target.max(dim=1).values

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=False)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
