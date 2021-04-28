"""
测试config

.. code-block:: bash

    python -m flame.ckcfg -c config/your_config.jsonnet

"""


import rich

from .argument import BasicArgs
from .config import config_snippet, from_snippet


def main():
    args = BasicArgs.from_args()
    snippet = config_snippet(args.local, args.config)
    cfg = from_snippet(snippet)
    rich.print(cfg)


if __name__ == '__main__':
    main()
