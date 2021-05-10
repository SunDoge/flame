local Mnist(root, train=true) = {
  _type: 'torchvision.datasets.MNIST',
  root: root,
  train: train,
  download: true,
};


local Transform = {
  _type: 'torchvision.transforms.Compose',
  transform_presets:: {
    to_tensor: {
      _type: 'torchvision.transforms.ToTensor',
    },
    normalize: {
      _type: 'torchvision.transforms.Normalize',
      mean: [0.1307],
      std: [0.3081],
    },
  },
  transform_list:: ['to_tensor', 'normalize'],
  transforms: [
    self.transform_presets[x]
    for x in self.transform_list
  ],
};


{
  train: {
    batch_size: 64,
    num_workers: 1,
    transform: Transform,
    dataset: Mnist(
      './data',
      train=true,
    ),
  },
  val: {
    batch_size: 64,
    num_workers: 1,
    transform: Transform,
    dataset: Mnist(
      './data',
      train=false,
    ),
  },
  engine: {
      _type: 'example.mnist.model.NetEngine',
  },
  max_epochs: 10,
}
