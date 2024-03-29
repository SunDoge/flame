from typing import Any, Callable
from flame.arguments import BaseArgs
from flame.distributed_training import start_distributed_training
from icecream import ic
from torchvision import transforms as T
from torch.utils.data import DataLoader
import flame
import torch

Args = BaseArgs


def main():
    args = Args.from_args(
        ['-c', 'example/mnist/pytorch_example_v3.jsonnet', '-d'])
    start_distributed_training(args)


def MnistTrain():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])


def main_worker(
    args: Args,
    train_loader: DataLoader,
    optimizer_fn: Callable[[Any], torch.optim.Optimizer],
    **kwargs
):
    ic(args)
    ic(kwargs)

    for image, label in train_loader:
        print(image.shape)
        print(label)
        break

    model = torch.nn.Linear(1, 2)
    optimizer = optimizer_fn(model.parameters())
    print(optimizer)


if __name__ == '__main__':
    main()
