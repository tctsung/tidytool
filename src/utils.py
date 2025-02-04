import logging
import os
from datetime import datetime
import pickle

#############################
# Internal helper functions #
#############################


def find_files(directory_path=".", file_extension=".json"):
    # TODO: find all files in a directory with specific file extension
    file_paths = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_extension):
                file_paths.append(os.path.join(root, file))
    return file_paths


def set_loggings(level=logging.INFO, func_name=""):
    """
    TODO: set logging levels for object by overwriting the root logger
    """
    if isinstance(level, str):
        level = level.upper()
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = log_levels[level]
    # Remove all handlers associated with the root logger object:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,  # set logging level
        format="----- %(levelname)s (%(asctime)s) ----- \n%(message)s\n",
    )  # set messsage format
    logging.critical(
        "Hello %s, The current logging level is: %s",
        func_name,
        logging.getLevelName(logging.getLogger().getEffectiveLevel()),
    )


def get_timestamp():
    current_timestamp = datetime.now()
    return current_timestamp.strftime("%Y-%m-%d %H:%M:%S")


def pickle_save(obj, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def pickle_load(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)
