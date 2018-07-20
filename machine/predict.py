#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

from nlp_tools import tokenizer
from machine import lsi, svm


def init_model(train_info):
    train_common_config = train_info['train_info']
    name = train_common_config['name']

    tfidf_config_name = train_info['train_info']['tfidf_config']

    train_config = train_info[name]
    tfidf_config = train_config[tfidf_config_name]

    num_topics = tfidf_config['num_topics']

    tfidf_model = lsi.recover_model(train_info)
    assert tfidf_model.lsi_model.num_topics == num_topics

    clf, le = svm.recover_svm_clf(train_info)

    return tfidf_model, clf, le


def predict(train_info, tfidf_model, clf, le, content):

    train_common_config = train_info['train_info']

    name = train_common_config['name']

    tfidf_config_name = train_info['train_info']['tfidf_config']
    ml_config_name = train_info['train_info']['classifier_config']

    train_config = train_info[name]
    tfidf_config = train_config[tfidf_config_name]
    ml_config = train_config[ml_config_name]
    prob = ml_config['is_prob']

    num_topics = tfidf_config['num_topics']

    doc = tokenizer.tokenizer([content],
                              is_keywords=False)

    corpus_tfidf = tfidf_model.compute_tfidf([doc[0]], is_file=False)

    corpus_lsi = tfidf_model.compute_lsi(corpus_tfidf, num_topics=num_topics,
                                         is_dense=True)

    if prob:
        pred = clf.predict_proba(corpus_lsi)
    else:
        pred = clf.predict(corpus_lsi)

    return pred
