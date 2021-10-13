import traceback

from .config import get_config
from .logger import close_logger, get_logger
from .main import _train
from .utils import get_argparse


def train(path_to_config: str) -> None:
    """Function to train NER model with exception handler.

    Args:
        path_to_config (str): Path to config.
    """

    # load config
    config = get_config(path_to_config=path_to_config)

    # get logger
    logger = get_logger()  # TODO: add path to save

    try:
        _train(
            config=config,
            logger=logger,
        )

    except:  # noqa
        close_logger(logger)
        print(traceback.format_exc())


def main() -> int:
    """Main function.

    Returns:
        int: Exit code.
    """

    # argument parser
    parser = get_argparse()
    args = parser.parse_args()

    # train
    train(path_to_config=args.path_to_config)

    return 0


if __name__ == "__main__":
    exit(main())
