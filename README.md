# flame


![CI workflow](https://github.com/SunDoge/flame/actions/workflows/python-package.yml/badge.svg)
![docs workflow](https://github.com/SunDoge/flame/actions/workflows/sphinx-make-html.yml/badge.svg)

## Attention
`pytorch` is not listed in the dependencies. You should install it manually.

## Install

### pip

If in China

```bash
pip install -U git+https://hub.fastgit.org/SunDoge/flame
```

else

```bash
pip install -U git+https://github.com/SunDoge/flame
```

### Manual

```bash
mkdir third_party
git submodule add https://hub.fastgit.org/SunDoge/flame third_party/flame
ln -s third_party/flame/flame ./
```

For projects with `flame` as submodule

```bash
git submodule update --init
```


## Usage

Docs: [https://sundoge.github.io/flame/](https://sundoge.github.io/flame/)

## Development

You should install [poetry](https://github.com/python-poetry/poetry) first. 

```bash
poetry install
```

## Core concepts

TODO
