#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

import logging
import os
import sys
from logging import Logger


def get_std_logger(
    name: str, path: str = "", level: str = "INFO", stdout: bool = False
) -> Logger:
    """Create a logger.

    Args:
        name (str): Name of the logger.
        path (str, optional): Path to the log file. If None, no log file is created. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.
        stdout (bool, optional): Whether to print the logs to stdout. Defaults to False.

    Returns:
        logging.Logger: The created logger.
    """
    # Check if the path exists
    os.makedirs(path, exist_ok=True)

    # Map string levels to logging constants
    level_mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    # Convert level string to logging constant
    if isinstance(level, str):
        level = level_mapping.get(level.upper(), logging.INFO)

    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    if path:
        log_path = os.path.join(path, name + ".log")
        file_handler = logging.FileHandler(filename=log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout:
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    return logger
