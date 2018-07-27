from pprint import pprint

import yaml

from machine import lsi, predict, train
# from project.keyword import keyword_filter_main
from utils.logger import logger
from utils.mysql_ctrl import MysqlCtrl


def predict_test(config_info, train_info):
    recommend_db = MysqlCtrl(config_info['recommend_mysql_r'])
    recommend_db.connect()

    select_sql = 'SELECT news_id, content FROM t_news_corpus_latest LIMIT 10;'

    ret, news = recommend_db.TB_select(select_sql)

    tfidf_model, clf, le = predict.init_model(train_info)

    for news_id, content in news:
        pred = predict.predict(train_info, tfidf_model, clf, le, content)
        pred = le.inverse_transform(pred)

        # 
        print(pred)

    recommend_db.close()


def main():

    log_file_path = './logs/train_sogou'
    logger.logger_init(log_file_path, stdout_level='info')

    train_yaml_path = './conf/train-sogou.yaml'
    config_yaml_path = './conf/config.yaml'

    with open(train_yaml_path, 'r') as f:
        train_info = yaml.load(f.read())

    with open(config_yaml_path, 'r') as f:
        config_info = yaml.load(f.read())

    try:

        # lsi.init_tfidf(train_info)

        # lsi.init_lsi_lda(train_info, recover_tfidf=True)

        # lsi.compute_lsi_lda(train_info, recover_m=False, recover_d=False, recover_tfidf=True)

        # train.train(train_info)

        predict_test(config_info, train_info)

        print('hello')
    except KeyboardInterrupt:

        logger.log.warn('shut down program')


if __name__ == '__main__':
    main()
