import os


def cache_dir_path(network_id, cache_name):
    cwd = os.getcwd()
    return os.path.join(cwd, '.cache', network_id, cache_name)
