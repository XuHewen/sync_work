from pprint import pprint

import yaml

from machine import lsi, predict, train
# from project.keyword import keyword_filter_main
from utils.logger import logger
from utils.mysql_ctrl import MysqlCtrl

from memory_profiler import profile
import time


def main():

    train_yaml_path = './conf/train-area.yaml'
    config_yaml_path = './conf/config.yaml'

    with open(train_yaml_path, 'r') as f:
        train_info = yaml.load(f.read())

    with open(config_yaml_path, 'r') as f:
        config_info = yaml.load(f.read())

    try:
        db_info = config_info['yff_mysql']

        recommend_db = MysqlCtrl(db_info)
        ret = recommend_db.connect()

        print('hello')
    except KeyboardInterrupt:
        logger.log.warn('shut down program')


if __name__ == '__main__':
    log_file_path = './logs/main'
    logger.logger_init(log_file_path, stdout_level='info')

    print('start ... ')
    for i in range(10):
        main()
    print('end ...')
    time.sleep(1000)
