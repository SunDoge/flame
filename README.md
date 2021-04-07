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

`flame`依赖几个核心概念：`Dependency Injection`，`Process`，`State`，`Events`。

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

