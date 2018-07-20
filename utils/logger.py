from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import logging
from datetime import datetime


class logger(object):
    log = logging

    @classmethod
    def logger_init(cls, log_file_name, stdout_level='info', fileout_level='warn', log_name=None):
        """get a logger

        """
        cls.log = logging.getLogger(name=log_name)
        cls.log.setLevel(logging.DEBUG)

        stdout_level = stdout_level.upper()
        fileout_level = fileout_level.upper()

        assert stdout_level in ['INFO', 'DEBUG', 'WARN', 'ERROR']
        assert fileout_level in ['INFO', 'DEBUG', 'WARN', 'ERROR']

        log_date = datetime.now().strftime("%Y-%m-%d")
        log_file_name = log_file_name + '-' + log_date + '.log'

        fh = logging.FileHandler(log_file_name)
        fh.setLevel(getattr(logging, fileout_level))

        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, stdout_level))

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        cls.log.addHandler(fh)
        cls.log.addHandler(ch)


if __name__ == '__main__':
    logger.logger_init('./logs/xhw')
    
