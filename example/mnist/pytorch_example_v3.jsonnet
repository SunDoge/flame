local Mnist(root, transform, train=true) = {
  _call: 'torchvision.datasets.MNIST',
  root: root,
  transform: transform,
  train: train,
  download: true,
};

{
    local root = self,
    _call: 'example.mnist.main_v2.main_worker',
    args: '$args',

    mnist_root:: './data/mnist',
    train_transform: {
        _call: 'example.mnist.main_v2.MnistTrain'
    },
    train_dataset: Mnist(
        root.mnist_root,
        '$train_transform',
        train=true
    ),
    train_loader: {
        _call: 'flame.helpers.create_data_loader',
        dataset: '$train_dataset',
        batch_size: 512,
        num_workers: 2,
    },
    optimizer_fn: {
        _use: 'torch.optim.SGD',
        lr: 0.1
    },
}
