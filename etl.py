import yaml

from ETL import clean_news_sogou, make_sogou_doc
from utils.logger import logger


def main():

    log_file_path = './logs/etl'
    logger.logger_init(log_file_path, stdout_level='info')

    train_yaml_path = './conf/train-sogou.yaml'
    config_yaml_path = './conf/config.yaml'

    with open(train_yaml_path, 'r') as f:
        train_info = yaml.load(f.read())

    with open(config_yaml_path, 'r') as f:
        config_info = yaml.load(f.read())

    make_sogou_doc.make_doc(train_info, config_info)


if __name__ == '__main__':
    main()
