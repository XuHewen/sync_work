#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import os
import sys

from w3lib.html import remove_tags

from nlp_tools import tokenizer
from utils.logger import logger
from utils.mysql_ctrl import MysqlCtrl


def _make_doc(name, text_list, doc_type, target, part_id,
              is_keyword=False, len_threshold=1,
              stop_words=None):

    if len(text_list) == 0:
        return []

    assert doc_type in ['train', 'test', 'predict']

    logger.log.info('tokenizing %d texts ...' % len(text_list))
    text_doc = tokenizer.tokenizer(
        text_list, is_keywords=is_keyword,
        stop_words=stop_words, len_threshold=len_threshold)

    # 将处理后的文档本地保存下来
    # 文档将保存在 ./data/gensim/original-corpus/doc_name 下面
    doc_dir = os.path.join('./data/gensim/{}'.format(name), 'original_corpus')
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)

    # 保存名字
    part_id = format(part_id, '04d')
    doc_name = '{0}-{1}-{2}-{3}.txt'.format(name,
                                            doc_type,
                                            target,
                                            part_id)

    doc_path = os.path.join(doc_dir, doc_name)

    logger.log.info('saving doc %s ...' % doc_path)
    with open(doc_path, 'w') as f:
        for doc in text_doc:
            doc = ','.join(doc)
            f.write(doc)
            f.write('\n')

    return text_doc


def load_news(recommend_db, target):
    select_sql = "SELECT content "\
                 "FROM t_sogou_news "\
                 "WHERE target = '%s' AND channel != 'C000008';" % target

    ret, news = recommend_db.TB_select(select_sql)
    if ret:
        logger.log.info('loaded %d news' % len(news))
        return news
    else:
        logger.log.error('load news error, exit')
        sys.exit(-1)


def make_doc(train_info, config_info):
    """ 从数据库导入原始文本，再分词或提取关键词，最后根据标签名称以txt保存在本地

    Arguments:
        train_info {dict/yaml} -- 有关训练和分词的参数配置
        config_info {dict/yaml} --  有关数据库等的参数配置
    """

    name = train_info['train_info']['name']

    logger.log.info('start making doc, %s ... ' % name)

    db_info = config_info['recommend_mysql_r']

    make_info = train_info['make_doc']

    len_threshold = make_info['len_threshold']
    is_keyword = make_info['is_keyword']

    tmp_dir = make_info.get('tmp_dir')  # jieba缓存目录，避免权限问题
    user_dict_path = make_info.get('user_dict_path')  # 导入用户词典，用于分词和提取关键词
    stop_words_path = make_info.get(
        'stop_words_path') if is_keyword else None  # 导入停用词，只能用于提取关键词
    idf_file_path = make_info.get(
        'idf_file_path') if is_keyword else None  # 导入idf词典，只能用于提取关键词

    tokenizer.jieba_init(tmp_dir=tmp_dir,
                         user_dict_path=user_dict_path,
                         stop_words_path=stop_words_path,
                         idf_file_path=idf_file_path)

    # 导入停用词用于过滤分词结果
    stop_words = tokenizer.get_stop_words(make_info.get('stop_words_path'))

    recommend_db = MysqlCtrl(db_info=db_info)
    ret = recommend_db.connect()
    if not ret:
        logger.log.error('connect to database error, exit')
        sys.exit(-1)

    select_sql = 'SELECT DISTINCT target FROM t_sogou_news;'

    ret, targets = recommend_db.TB_select(select_sql)

    targets = [x[0] for x in targets]

    doc_type = 'train'
    part_id = 1
    for tag in targets:
        news = load_news(recommend_db, tag)
        if len(news) > 0:
            text_list = [item[-1] for item in news if len(item[-1]) > 200]
            _make_doc(name, text_list, doc_type, tag, part_id,
                    stop_words=stop_words, is_keyword=is_keyword,
                    len_threshold=len_threshold)
