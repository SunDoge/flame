# flame


![CI workflow](https://github.com/SunDoge/flame/actions/workflows/python-package.yml/badge.svg)
![docs workflow](https://github.com/SunDoge/flame/actions/workflows/sphinx-make-html.yml/badge.svg)

## Attention
`pytorch` is not listed in the dependencies. You should install it manually.

## Install

If in China

```bash
pip install -U git+https://hub.fastgit.org/SunDoge/flame
```

else

```bash
pip install -U git+https://github.com/SunDoge/flame
```

## Usage

Docs: [https://sundoge.github.io/flame/](https://sundoge.github.io/flame/)

## Development

You should install [poetry](https://github.com/python-poetry/poetry) first. 

```bash
poetry install
```

## Core concepts

`flame`依赖几个核心概念：`Dependency Injection`，`Process`，`State`，`Engine`。

### Dependency Injection

依赖注入贯穿整个框架。`flame`使用的依赖注入框架是[injector](https://github.com/alecthomas/injector)。

一个简单的依赖注入的example如下

```python
from injector import inject, singleton, Injector, Module, provider
from typing import NewType

# 我们可以利用typing中的NewType为dict构建alias
Config = NewType('Config', dict)
Args = NewType('Args', dict)

class MyModule(Module):
    
    @singleton # 说明这个函数返回的对象是一个单例
    @provider # 说明这个函数会提供一个对象，对象的类型是函数的返回值
    def configure_args(self) -> Args:
        args = {'batch_size': 16}
        return args

    
    @singleton
    @provider # 这里args的类型标注为Args，由于上面定义过Args的构造方式，这里会自动注入单例args
    def configure_config(self, args: Args) -> Config:
        batch_size = args['batch_size']
        cfg = {'train_batch_size': batch_size, 'val_batch_size': batch_size}
        return cfg


@inject # 如果是一个类需要自动注入，使用inject而不是provider
class A:

    def __init__(self, cfg: Config):
        self.cfg = cfg

container = Injector(MyModule) # injector可能和包名冲突，所以我们一般把dependency injector叫成container
cfg = container.get(Config) # Config在MyModule里面定义了绑定的对象，可以通过get方法得到

a = container.get(A) # 不是alias的类也能根据__init__中标注的类型自动构建

print(a.cfg)
```

输出
```text
{'train_batch_size': 16, 'val_batch_size': 16}
```

依赖注入有几个好处：

1. 方便测试。函数或类需要声明自己依赖的对象，测试时只需要构造确定数量的对象。
2. 方便修改。如果不使用依赖注入，修改一个函数的签名的同时，需要修改所有调用该函数的地方。使用依赖注入后，只需要修改函数签名。

### Process

`Process`是一个训练最核心的部分。一个基本的监督训练的process如下

```python
from flame.pytorch.typing_prelude import * # 这里面定义了很多有用的类型
from injector import inject
import torch

class SupervisedProcess:

    @inject # 这里用到的东西都可以自动注入
    def __init__(self, model: Model, criterion: Criterion, optimizer: Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer


    def training_step(self, batch):
        # 训练过程包括forward和backward，不包括update optimizer，这一步可以定义为模板
        output = self.forward_step(batch)
        loss = output['loss']
        loss.backward()

        return output

    def validation_step(self, batch):
        # 推理过程不需要用到loss，只需要返回output，这一步可以定义为模板
        with torch.no_grad():
            output = self.forward_step(batch)
        
        return output

    def forward_step(self, batch):
        # 推理过程是用户需要重写的部分，包括如何解析一个batch的数据，如何将数据传给模型，如何计算loss并返回，output包括哪些内容
        image, label = batch
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        pred = self.model(image)

        loss = criterion(pred, label)

        # 在forward step需要计算出batch size，后续有很多步骤需要batch size
        batch_size = image.size(0)

        return self.output(loss=loss, batch_size=batch_size, pred=pred, label=label)


    def update(self):
        # 如何更新weight

        self.optimizer.step()
        self.optimizer.zero_grad() # zero_grad放在step后是为了兼容gradient accumulation

    def train(self, mode=True):
        # 定义如何切换model的训练/推理状态
        self.model.train(mode=mode)
```

在训练的时候，一般调用顺序为

```python
# At epoch_started
supervised_process.train()

# At iteration
for batch in dataloader:
    output = supervised_process.training_step(batch)

    supervised_process.update()

```

在推理的时候，一般调用顺序为

```python
# At epoch_started
supervised_process.eval()

# At iteration
for batch in dataloader:
    output = supervised_process.validation_step(batch)
```

### State

在模型的训练过程中，我们需要维护很多状态，比如当前的iteration和epoch，总的iteration和epoch，每个batch的accuracy和loss，最好的accuracy或loss。为了便于维护，个人建议把所有的状态放在一起，这样在保存状态的时候也会方便不少。

目前用到的`State`如下

### Engine

pass
