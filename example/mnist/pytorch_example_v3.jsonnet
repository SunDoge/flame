local Mnist(root, transform, train=true) = {
  _name: 'torchvision.datasets.MNIST',
  root: root,
  transform: transform,
  train: train,
  download: true,
};

{
    local root = self,
    _name: 'example.mnist.main_v2.main_worker',
    args: '$args',

    mnist_root:: './data/mnist',
    train_transform: {
        _name: 'example.mnist.main_v2.MnistTrain'
    },
    train_dataset: Mnist(
        root.mnist_root,
        '$train_transform',
        train=true
    ),
    train_loader: {
        _name: 'flame.helpers.create_data_loader',
        dataset: '$train_dataset',
        batch_size: 512,
        num_workers: 2,
    },
}
