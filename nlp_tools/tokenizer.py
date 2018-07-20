#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import multiprocessing
import os

import jieba
import jieba.analyse
import jieba.posseg as pseg

from utils.logger import logger
from utils.wrapper import time_consumed


def jieba_init(tmp_dir=None, user_dict_path=None,
               stop_words_path=None, idf_file_path=None):

    if tmp_dir:
        if not os.path.exists(tmp_dir):
            logger.log.warn(
                'provided jieba cache dir not exists, %s' % tmp_dir)
        else:
            jieba.dt.tmp_dir = tmp_dir
            logger.log.info('set jieba cache dir, %s' % tmp_dir)

    if user_dict_path:
        if not os.path.exists(user_dict_path):
            logger.log.warn('can not find user dict file, %s' % user_dict_path)
        else:
            jieba.load_userdict(user_dict_path)
            logger.log.info('load user dict, %s' % user_dict_path)

    if stop_words_path:
        if not os.path.exists(stop_words_path):
            logger.log.warn('can not find stop words file, %s' %
                            stop_words_path)
        else:
            jieba.analyse.set_stop_words(stop_words_path)
            logger.log.info('load stop words, %s' % stop_words_path)

    if idf_file_path:
        if not os.path.exists(idf_file_path):
            logger.log.warn('can not find idf file, %s' % idf_file_path)
        else:
            jieba.analyse.set_idf_path(idf_file_path)
            logger.log.info('load idf file, %s' % idf_file_path)


def get_stop_words(stop_words_path):
    if not os.path.exists(stop_words_path):
        logger.log.erroe('can not find stop words file, %s' % stop_words_path)
        return []
    else:
        stop_words = set()
        with open(stop_words_path, 'r') as f:
            for word in f:
                stop_words.add(word.strip('\n').strip())

        stop_words.add('\n')
        stop_words.add(' ')
        stop_words.add('\u3000')

        return stop_words


def _tokenizer(sub_text_list, is_keyword,
               num_keywords, with_weight, stop_words, len_threshold):

    sub_doc_list = []
    if is_keyword:
        for text in sub_text_list:
            text_keywords = jieba.analyse.extract_tags(
                text, num_keywords, withWeight=with_weight)
            sub_doc_list.append(text_keywords)
    else:
        for text in sub_text_list:
            text_words = jieba.cut(text)
            temp = []
            if stop_words:
                for word in text_words:
                    word = word.strip('\n').strip()
                    if word not in stop_words and len(word) > len_threshold:
                        temp.append(word)
            else:
                temp = list(text_words)
            sub_doc_list.append(temp)

    return sub_doc_list


# @time_consumed
def tokenizer(text_list, num_process=4, is_keywords=False,
              num_keywords=20, with_weight=False,
              stop_words=None, len_threshold=0):

    if not isinstance(text_list, list):
        raise TypeError('text list should be a list')

    if len(text_list) < 4:
        # print('process 0: 0 => %d: %d' % (len(text_list)-1, len(text_list)))
        return _tokenizer(text_list, is_keywords,
                          num_keywords, with_weight, stop_words, len_threshold)

    pool = multiprocessing.Pool(processes=num_process)

    doc_list = []
    res = []

    step_size = len(text_list) // num_process
    for i in range(num_process):
        start = step_size * i
        end = step_size * (i + 1)
        if i == num_process - 1:
            end = len(text_list)

        # print('process %d: %d => %d: %d' % (i, start, end, end - start))

        res.append(pool.apply_async(
            _tokenizer, (text_list[start:end], is_keywords,
                         num_keywords, with_weight, stop_words, len_threshold)))

    pool.close()
    pool.join()

    for sub_doc_list in res:
        doc_list.extend(sub_doc_list.get())

    assert len(doc_list) == len(text_list)

    return doc_list
