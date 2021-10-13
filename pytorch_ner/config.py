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

    return config
