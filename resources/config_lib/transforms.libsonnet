local torchvision_prefix = 'torchvision.transforms.';


{
  Compose(keys, lib, type):: {
    _type: type,
    transforms: [
      lib[key]
      for key in keys
    ],
  },

  torchvision_transforms: {
    Compose(keys, lib):: $.Compose(keys, lib, torchvision_prefix + 'Compose'),

    RandomResizedCrop(size=224):: {
      _type: torchvision_prefix + 'RandomResizedCrop',
      size: size,
    },

    RandomHorizontalFlip():: {
      _type: torchvision_prefix + 'RandomHorizontalFlip',
    },
  },
}
