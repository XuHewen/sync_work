#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

from nlp_tools import tokenizer
from machine import lsi, svm, logistic
from machine.get_project_config import get_project_config


def init_model(project_info):

    _, _, _, topic_config, train_method, _ = get_project_config(project_info)
    num_topics = topic_config['num_topics']

    tfidf_model = lsi.recover_model(project_info)
    assert tfidf_model.lsi_model.num_topics == num_topics

    if train_method == 'logistic':
        clf, le = logistic.recover_clf(project_info)
    elif train_method == 'svm':
        clf, le = svm.recover_clf(project_info)

    return tfidf_model, clf, le


def predict(project_info, tfidf_model, clf, le, content):

    _, _, _, topic_config, train_method, train_config = get_project_config(project_info)
    
    prob = train_config.get('probability', False)

    num_topics = topic_config['num_topics']

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
