local flame = import '../../jsonnet/flame.libsonnet';

local Mnist(root, transform, train=true) = {
  _call: 'torchvision.datasets.MNIST',
  root: root,
  transform: transform,
  train: train,
  download: true,
};

{
  local root = self,
  _call: 'example.mnist.main_v3.Trainer',
  args: '$args',
  batch_size: 128,
  num_workers: 2,
  max_epochs: 14,
  log_interval: 10,

  mnist_root:: './data/mnist',
  train_transform: {
    _call: 'example.mnist.presets.MnistTransform',
  },
  train_dataset: Mnist(
    root.mnist_root,
    '$train_transform',
    train=true
  ),
  train_loader: {
    _call: 'flame.helpers.create_data_loader',
    dataset: '$train_dataset',
    batch_size: root.batch_size,
    num_workers: root.num_workers,
  },
  test_transform: root.train_transform,
  test_dataset: Mnist(
    root.mnist_root,
    '$test_transform',
    train=false,
  ),
  test_loader: {
    _call: 'flame.helpers.create_data_loader',
    dataset: '$test_dataset',
    batch_size: root.batch_size * 10,
    num_workers: root.num_workers,
  },
  data_module: {
    _call: 'flame.experimental.trainer.DataModule',
    train_loader: '$train_loader',
    test_loader: '$test_loader',
  },
  optimizer_fn: {
    _use: 'torch.optim.Adadelta',
    lr: flame.normLr(4.0, root.batch_size),
  },
  scheduler_fn: {
    _use: 'torch.optim.lr_scheduler.StepLR',
    gamma: 0.7,
    step_size: 1,
  },
  model: {
    _call: 'example.mnist.model.Net'
  },
}
