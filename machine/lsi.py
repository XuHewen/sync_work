#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import os
import sys
from glob import glob

from machine.get_project_config import get_project_config
from nlp_tools import gensim_tfidf
from utils.logger import logger


def init_tfidf(project_info):

    logger.log.info('start initial tfidf model ... ')

    name, dict_config, _, _, _, _ = get_project_config(project_info)

    tfidf_model = gensim_tfidf.GensimTfidf(name=name)

    doc_dir = './data/gensim/{0}/original_corpus'.format(name)
    # doc_dir = os.path.join(base_dir, name)

    if not os.path.exists(doc_dir):
        logger.log.error('doc dir %s not exists, exit!' % doc_dir)
        sys.exit(-1)

    pattern = './data/gensim/{0}/original_corpus/{0}-train-*-*.txt'.format(
        name)
    doc_path = glob(pattern)

    # 文本迭代器，减少内存消耗
    my_doc = gensim_tfidf.MyDoc(doc_path)
    # 初始化词典，并保存
    tfidf_model.add_document_from_file(
        my_doc, is_save=True, **dict_config)
    # 初始化 tfidf model, 并保存
    tfidf_model.init_tfidf_from_file(my_doc, is_save=True)
    # 计算 tfidf 用于初始化 lsi model, 并保存
    tfidf_model.compute_tfidf(my_doc, is_save=True, corpus_name=name)

    return tfidf_model


def init_lsi_lda(project_info, recover_tfidf=True):

    logger.log.info('start init lsi model ... ')

    name, dict_config, topic_method, topic_config, _, _ = get_project_config(project_info)

    assert topic_method in ['lsi', 'lda']

    if topic_method == 'lsi':
        if recover_tfidf:
            model_dir = './data/gensim/{}/model'.format(name)
            model_name = '{}-tfidf.model'.format(name)
            tfidf_model_path = os.path.join(model_dir, model_name)
            if not os.path.exists(tfidf_model_path):
                logger.log.error('tfidf model %s not eixsts, exit' %
                                tfidf_model_path)
                sys.exit(-1)
            else:
                tfidf_model = gensim_tfidf.GensimTfidf(name=name)
                tfidf_model.load_dictionary()
                tfidf_model.load_tfidf_model()
        else:
            tfidf_model = init_tfidf(project_info)

        corpus_tfidf = tfidf_model.load_tfidf_corpus(name)

        tfidf_model.init_lsi_model(
            corpus_tfidf, is_save=True, **topic_config)

    elif topic_method == 'lda':

        doc_dir = './data/gensim/{0}/original_corpus'.format(name)
        if not os.path.exists(doc_dir):
            logger.log.error('doc dir %s not exists, exit!' % doc_dir)
            sys.exit(-1)

        pattern = './data/gensim/{0}/original_corpus/{0}-train-*-*.txt'.format(name)
        doc_path = glob(pattern)

        # 文本迭代器，减少内存消耗
        my_doc = gensim_tfidf.MyDoc(doc_path)
        
        tfidf_model = gensim_tfidf.GensimTfidf(name=name)
        if recover_tfidf:
            tfidf_model.load_dictionary()
        else:
            tfidf_model.add_document_from_file(my_doc, is_save=True, **dict_config)

        my_corpus = gensim_tfidf.MyCorpus(gensim_tfidf.MyDoc(doc_path), tfidf_model.dictionary)

        tfidf_model.init_lda_model(my_corpus, is_save=True, **topic_config)

    return tfidf_model


def recover_model(project_info):
    logger.log.info('start recovering lsi model ... ')

    name, _, topic_method, _, _, _ = get_project_config(project_info)

    assert topic_method in ['lsi', 'lda']

    tfidf_model = gensim_tfidf.GensimTfidf(name=name)
    tfidf_model.load_dictionary()
    tfidf_model.load_tfidf_model()

    if topic_method == 'lsi':
        tfidf_model.load_lsi_model()
    elif topic_method == 'lda':
        tfidf_model.load_lda_model()

    return tfidf_model


def compute_lsi_lda(project_info, recover_m=True,
                    recover_d=True, recover_tfidf=False):

    name, dict_config, topic_method, topic_config, _, _ = get_project_config(project_info)

    pattern = './data/gensim/{0}/original_corpus/{0}-train-*-*.txt'.format(
        name)
    doc_path = glob(pattern)

    data = []

    num_topics = topic_config['num_topics']

    if recover_d:
        tfidf_model = gensim_tfidf.GensimTfidf(name=name)
        
        doc_path = [x.split('/')[-1] for x in doc_path]

        for doc in doc_path:
            _, _, tag, _ = doc[:-4].split('-')

            if topic_method == 'lsi':
                corpus_lsi = tfidf_model.load_lsi_corpus(doc[:-4], is_npy=True)
                assert corpus_lsi.shape[1] == num_topics
                data.append((corpus_lsi, tag))

            elif topic_method == 'lda':
                corpus_lda = tfidf_model.load_lda_corpus(doc[:-4], is_npy=True)
                assert corpus_lsi.shape[1] == num_topics
                data.append((corpus_lda, tag))

        return data

    if recover_m:
        tfidf_model = recover_model(project_info)
        if topic_method == 'lsi':
            assert tfidf_model.lsi_model.num_topics == num_topics
        elif topic_method == 'lda':
            assert tfidf_model.lda_model.num_topics == num_topics
    else:
        tfidf_model = init_lsi_lda(project_info, recover_tfidf)

    for doc in doc_path:
        _, _, tag, _ = doc[:-4].split('-')
        corpus_name = doc.split('/')[-1][:-4]
        my_doc = gensim_tfidf.MyDoc(doc)

        if topic_method == 'lsi':
            corpus_tfidf = tfidf_model.compute_tfidf(my_doc)
            corpus_lsi = tfidf_model.compute_lsi(corpus_tfidf,
                                                 num_topics=num_topics,
                                                 is_dense=True,
                                                 is_npy_save=True,
                                                 corpus_name=corpus_name)
            data.append((corpus_lsi, tag))

        elif topic_method == 'lda':
            my_corpus = gensim_tfidf.MyCorpus(gensim_tfidf.MyDoc(doc), tfidf_model.dictionary)
            corpus_lda = tfidf_model.compute_lda(my_corpus,
                                                 num_topics=num_topics,
                                                 is_dense=True,
                                                 is_npy_save=True,
                                                 corpus_name=corpus_name)
            data.append((corpus_lda, tag))

    return data
