from .main import _train
from .utils import get_argparse

if __name__ == "__main__":

    # argument parser
    parser = get_argparse()
    args = parser.parse_args()

    _train(path_to_config=args.path_to_config)
