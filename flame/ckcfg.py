"""
测试config::

    python -m flame.ckcfg -c config/your_config.jsonnet

"""


from .argument import parse_basic_args, BasicArgs
from .config import config_snippet, from_snippet
import rich


def main():
    args = parse_basic_args()
    snippet = config_snippet(args.local, args.config)
    cfg = from_snippet(snippet)
    rich.print(cfg)



if __name__ == '__main__':
    main()