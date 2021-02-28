import json
import os

from typing import Union, List, Dict


def read_json(filepath: str) -> Union[Dict, List]:
    """
    Read data from a json file.

    :param filepath: Path to the json file to read
    :return: Data from the json file
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def write_json(filepath: str, data: Union[List, Dict]) -> None:
    """
    Write data into a json file

    :param filepath: Path to the json file
    :param data: Data to write
    :return:
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w+') as f:
        json.dump(data, f)
