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
  // engine: {
  //   _type: 'example.mnist.model.NetEngine',
  //   cfg: {
  //     max_epochs: $.max_epochs,
  //     print_freq: 10,
  //   },
  // },
  engine: 'example.mnist.classification_engine.ClassificationEngine',

  engine_cfg: {
    _type: 'flame.pytorch.experimental.compact_engine.engine_v3.BaseEngineConfig',
    max_epochs: 10,
  },

  model: {
    _type: 'example.mnist.model.Net',
  },
  criterion: {
    // TODO
    _type: 'torch.nn.CrossEntropyLoss',
  },
  optimizer: {
    // TODO
    _type: 'torch.optim.Adadelta',
    lr: 1.0,
  },
  scheduler: {
    _type: 'torch.optim.lr_scheduler.StepLR',
    step_size: 1,
    gamma: 0.7,
  },
}
