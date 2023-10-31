# -*- coding: utf-8 -*-

import logging
import json
from collections import OrderedDict
import logging.config
from pathlib import Path

log_levels = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def setup_logging(save_dir, log_config='logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(__file__).parent.joinpath(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)


def get_logger(name, verbosity=2):
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                   log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger
