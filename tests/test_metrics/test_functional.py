# from flame.pytorch.experimental.metrics import functional as MF
from flame.pytorch.metrics import functional as MF
import torch


def test_topk_acc():
    output = torch.eye(10)
    target_100 = torch.arange(10)
    acc1, acc5 = MF.topk_accuracy(output, target_100, topk=(1, 5))
    assert acc1.item() == 100.
    assert acc5.item() == 100.

    jit_topk_accuracy = torch.jit.script(MF.topk_accuracy)

    acc1, acc5 = jit_topk_accuracy(output, target_100, topk=(1, 5))
    assert acc1.item() == 100.
    assert acc5.item() == 100.
    