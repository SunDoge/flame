"""
测试config

.. code-block:: bash

    python -m flame.ckcfg -c config/your_config.jsonnet

"""


import rich

from .arguments import BaseArgs


def main():
    args = BaseArgs.from_args()
    config = args.parse_config()
    rich.print(config)


if __name__ == '__main__':
    main()
