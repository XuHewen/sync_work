import yaml

import sys
from ETL import make_business_doc
from utils.logger import logger
from utils.mysql_ctrl import MysqlCtrl


def main():

    log_file_path = './logs/etl'
    logger.logger_init(log_file_path, stdout_level='info')

    train_yaml_path = './conf/train-business.yaml'
    config_yaml_path = './conf/config.yaml'

    with open(train_yaml_path, 'r') as f:
        project_info = yaml.load(f.read())

    with open(config_yaml_path, 'r') as f:
        config_info = yaml.load(f.read())

    db_info = config_info['recommend_mysql_r']
    recommend_db = MysqlCtrl(db_info=db_info)
    ret = recommend_db.connect()
    if not ret:
        logger.log.error('connect to database error, exit')
        sys.exit(-1)

    # train-sogou.yaml
    # clean_news_sogou2.load_sogou_news(recommend_db)
    make_business_doc.make_doc(project_info, config_info)

    # train-daily.yaml
    # make_daily_doc.make_doc(project_info, config_info)

    # recommend_db.close()

if __name__ == '__main__':
    main()
