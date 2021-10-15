import logging
import sys
from typing import Optional


def get_logger(path_to_logfile: Optional[str] = None) -> logging.Logger:
    """Get logger.

    Args:
        path_to_logfile (Optional[str], optional): Path to logfile. Defaults to None.

    Returns:
        logging.Logger: Logger.
    """

    logger = logging.getLogger("pytorch-ner-train")
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)

    if path_to_logfile:
        file_handler = logging.FileHandler(path_to_logfile)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Close logger.
    Source: https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile

    Args:
        logger (logging.Logger): Logger.
    """

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
