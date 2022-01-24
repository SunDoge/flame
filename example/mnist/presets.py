"""
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torchvision.transforms as T


def MnistTransform():
    return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
