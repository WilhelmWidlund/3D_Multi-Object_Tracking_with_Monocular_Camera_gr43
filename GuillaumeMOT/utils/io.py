import os
from typing import Iterable, IO, Optional
# ------------ Altered code --------------------------------------------
from os import path, makedirs
from ujson import dump
# ------------ End altered code ----------------------------------------


def makedirs_if_new(path: str) -> bool:
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def close_files(files: Iterable[Optional[IO]]) -> None:
    for f in files:
        if f is not None:
            f.close()


def create_writable_file_if_new(folder: str, name: str):
    makedirs_if_new(folder)
    results_file = os.path.join(folder, name + '.txt')
    if os.path.isfile(results_file):
        return None
    return open(results_file, 'w')


# ------------ Altered code --------------------------------------------
def store_dataset_info(save_path: str, info: dict):
    if not path.exists(save_path):
        makedirs(save_path)
    info_file = path.join(save_path, "dataset_info.json")
    with open(info_file, 'w') as f:
        dump(info, f, indent=4)
# ------------ End altered code ----------------------------------------
