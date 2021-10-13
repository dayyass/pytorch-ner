from argparse import ArgumentParser

from .main import _train


def get_argparse() -> ArgumentParser:
    """Get argument parser.

    Returns:
        ArgumentParser: Argument parser.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        help="Path to config",
    )

    return parser


if __name__ == "__main__":

    # argument parser
    parser = get_argparse()
    args = parser.parse_args()

    _train(path_to_config=args.path_to_config)
