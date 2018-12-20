import os


def cache_dir_path(network_id, cache_name):
    return os.path.join('.cache', network_id, cache_name)
