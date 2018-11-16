import os
import pathlib
from shutil import copyfile
import gzip

import numpy as np

tmp_file_extension = 'tmp'


def append_data(data, dir_path, file_name):
    _prepare_tmp_file(dir_path, file_name)
    _append_into_tmp_file(data, dir_path, file_name)
    _replace_with_tmp_files(dir_path, file_name)


def load_data(dir_path, file_name):
    file_path = _file_path(dir_path, file_name)
    if not os.path.isfile(file_path):
        return None

    with _open_file(file_path, 'rb') as file:
        data = []
        while True:
            try:
                data.append(np.load(file))
            except OSError:
                break
    return data


def _prepare_tmp_file(dir_path, file_name):
    file_path = _file_path(dir_path, file_name)
    tmp_file_path = _tmp_file_path(dir_path, file_name)
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    if os.path.isfile(file_path):
        copyfile(file_path, tmp_file_path)


# noinspection PyTypeChecker
def _append_into_tmp_file(data, dir_path, file_name):
    tmp_file_path = _tmp_file_path(dir_path, file_name)
    with _open_file(tmp_file_path, 'ab') as temp_file:
        for row in data:
            np.save(temp_file, np.asarray(row))


def _replace_with_tmp_files(dir_path, file_name):
    file_path = _file_path(dir_path, file_name)
    tmp_file_path = _tmp_file_path(dir_path, file_name)
    os.rename(tmp_file_path, file_path)


def _file_path(dir_path, file_name):
    return os.path.join(dir_path, file_name)


def _tmp_file_path(dir_path, file_name):
    return os.path.join(dir_path, '{}.{}'.format(file_name, tmp_file_extension))


def _open_file(file_path, mode):
    # TODO: Choose one of these
    # return open(file_path, mode)
    return gzip.GzipFile(file_path, mode)
