{
  train: {
    local this = self,
    transform:: {
      _path: 'xxxx',
    },
    dataset:: {
      _path: 'xxx',
      transform: this.transform,
    },
    data_loader: {
      _path: 'flame.pytorch.helpers.create_data_loader',
      dataset: this.dataset,
      batch_size: 128,
      num_workers: std.floor(self.batch_size / 16),
    },
  },
  val: {
    data_loader: {

    },
  },
}
