from pprint import pprint

import yaml

from machine import lsi, predict, train
# from project.keyword import keyword_filter_main
from utils.logger import logger
from utils.mysql_ctrl import MysqlCtrl
import numpy as np
import jieba.analyse
from nlp_tools import keyword_filter


def titleContentKeywordMatch(news_id, title, content):
    """检测新闻正文是否包含文章关键字

    Arguments:
        title {str} -- 新闻标题
        content {str} -- 新闻正文
    """
    titleKeywords = jieba.analyse.extract_tags(title, 10)

    checker = keyword_filter.TrieCheck()

    for word in titleKeywords:
        checker.add_keyword(word)

    res = checker.get_keyword(content)

    if not res:
        print(news_id, title)



def predict_test(config_info):
    recommend_db = MysqlCtrl(config_info['recommend_mysql_r'])
    recommend_db.connect()

    select_sql = 'SELECT news_id, info_title, content FROM t_news_corpus_latest;'

    ret, news = recommend_db.TB_select(select_sql)

    res = []
    for news_id, title, content in news:
        titleContentKeywordMatch(news_id, title, content)

    # insert_sql = 'INSERT INTO t_result '\
    #              '(news_id, title, pred, content) '\
    #              'VALUES (%s, %s, %s, %s);'

    # recommend_db.TB_insert(insert_sql, res)

    recommend_db.close()


def main():

    log_file_path = './logs/test'
    logger.logger_init(log_file_path, stdout_level='info')

    train_yaml_path = './conf/train-sogou.yaml'
    config_yaml_path = './conf/config.yaml'

    with open(train_yaml_path, 'r') as f:
        project_info = yaml.load(f.read())

    with open(config_yaml_path, 'r') as f:
        config_info = yaml.load(f.read())

    try:


        predict_test(config_info)

        # print('hello')
    except KeyboardInterrupt:

        logger.log.warn('shut down program')


if __name__ == '__main__':
    main()