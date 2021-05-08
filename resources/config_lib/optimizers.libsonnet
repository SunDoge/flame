local prefix = 'torch.optim.';

{
  SGD(lr=0.1):: {
    _type: prefix + 'SGD',
    lr: lr,
    momentum: 0.9,
    dampening: 0.0,
    weight_decay: 1e-4,
    nesterov: false,
  },
  Adam(lr=1e-4):: {
    _type: prefix + 'Adam',
    lr: lr,
    betas: [0.9, 0.999],
    eps: 1e-8,
    weight_decay: 0.0,
    amsgrad: false,
  },
}
