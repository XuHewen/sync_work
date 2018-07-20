#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import os
import pickle
import random
import sys
from datetime import datetime

from utils import mysql_ctrl
from utils.cfg_parser import CfgParser
from utils.logger import logger


def clean_news():
    """reading news from txt file to fixed format
    """
    res = []
    corpus_dir_path = './data/business-corpus/2017'
    news_dir_names = [x for x in os.listdir(
        corpus_dir_path) if x.startswith('201')]
    news_dir_paths = [os.path.join(corpus_dir_path, x) for x in news_dir_names]

    for i, temp in enumerate(news_dir_paths):
        news_paths = [os.path.join(temp, x)
                      for x in os.listdir(temp) if x.endswith('.txt')]
        news_date = datetime.strptime(news_dir_names[i], '%Y%m%d')

        if len(news_paths) > 0:
            for news in news_paths:
                with open(news, 'r') as f:
                    title = f.readline()
                    title = title.strip('\n').strip()
                    content = f.read()
                    content = content.strip('\n').strip()

                if title:
                    res.append((title, content, news_date, 0))

    return res


def load_neg_news(recommend_db, size):
    with open('./data/keyword/temp.pkl', 'rb') as f:
        news_list = pickle.load(f)

    if size > len(news_list):
        size = len(news_list)

    sample_list = random.sample(news_list, size)

    return sample_list
    

def insert_news(recommend_db, news_list):
    """insert news to mysql
    """
    insert_sql = "INSERT IGNORE INTO t_business_news "\
                 "(title, content, send_date, target) "\
                 "VALUES (%s, %s, %s, %s);"

    ret = recommend_db.TB_insert(insert_sql, news_list)
    if not ret:
        logger.log.error('insert cleaned news error!')
    return ret


def main():

    log_file_path = './logs/main'
    logger.logger_init(log_file_path)

    cfg_file_path = './conf/config.ini'
    cfg_parser = CfgParser(cfg_file_path)

    db_info = cfg_parser.get_cfg_dict('recommend_mysql_r')
    recommend_db = mysql_ctrl.MysqlCtrl(db_info=db_info)
    ret = recommend_db.connect()
    if not ret:
        logger.log.error('connect to database error, exit')
        sys.exit(-1)

    news_list = clean_news()
    # news_list = load_neg_news(recommend_db, 3000)
    ret = insert_news(recommend_db, news_list)

    recommend_db.close()


if __name__ == '__main__':
    main()
