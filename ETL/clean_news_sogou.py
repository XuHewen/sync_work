#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import os
import sys

import yaml

from utils.logger import logger
from utils.mysql_ctrl import MysqlCtrl


def load_sogou_news():
    """reading news from txt file to fixed format
    """
    res = []
    corpus_dir_path = './data/sogou'
    news_dir_names = [x for x in os.listdir(
        corpus_dir_path) if x.startswith('C')]

    news_dir_paths = [os.path.join(corpus_dir_path, x) for x in news_dir_names]

    for i, temp in enumerate(news_dir_paths):

        news_paths = [os.path.join(temp, x)
                      for x in os.listdir(temp) if x.endswith('.txt')]

        channel = news_dir_names[i]

        if len(news_paths) > 0:
            for news in news_paths:
                with open(news, 'rb') as f:
                    content = f.read()
                    content = content.decode('gbk', errors='ignore')
                    content = content.replace('&nbsp;', '').replace('&nbsp', '')

                res.append((channel, content, 'sogou'))

    return res


def load_yff_news(recommend_db):
    select_sql = 'SELECT content FROM t_news_corpus_latest LIMIT 15000;'
    ret, news = recommend_db.TB_select(select_sql)
    res = []
    for i in news:
        res.append(('yff', i, 'yff'))

    return res


def insert_news(recommend_db, news_list):
    """insert news to mysql
    """
    insert_sql = "INSERT IGNORE INTO t_sogou_news "\
                 "(channel, content, target) "\
                 "VALUES (%s, %s, %s);"

    ret = recommend_db.TB_insert(insert_sql, news_list)
    if not ret:
        logger.log.error('insert cleaned news error!')
    return ret


def main():

    log_file_path = './logs/clean_news'
    logger.logger_init(log_file_path)

    cfg_file_path = './conf/config.yaml'
    with open(cfg_file_path, 'r') as f:
        cfg_info = yaml.load(f.read())

    db_info = cfg_info['recommend_mysql_r']
    recommend_db = MysqlCtrl(db_info=db_info)
    ret = recommend_db.connect()
    if not ret:
        logger.log.error('connect to database error, exit')
        sys.exit(-1)

    # news_list = load_sogou_news()
    news_list = load_yff_news(recommend_db)
    
    ret = insert_news(recommend_db, news_list)

    recommend_db.close()


if __name__ == '__main__':
    main()
