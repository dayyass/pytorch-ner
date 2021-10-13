import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def get_config(path_to_config: str) -> Dict[str, Any]:
    """Get config.

    Args:
        path_to_config (str): Path to config.

    Returns:
        Dict[str, Any]: Config.
    """

    with open(path_to_config, mode="r") as fp:
        config = yaml.safe_load(fp)

    config["save"]["path_to_folder"] = (
        Path(config["save"])
        / f"model_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
    )

    # mkdir if not exists
    config["save"]["path_to_folder"].absolute().mkdir(parents=True, exist_ok=True)

    return config
