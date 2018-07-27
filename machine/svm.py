#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from utils.logger import logger


def svm_clf(train_info, data, target, le, is_save=True):

    train_common_config = train_info['train_info']

    name = train_common_config['name']

    tfidf_config_name = train_info['train_info']['tfidf_config']
    ml_config_name = train_info['train_info']['classifier_config']
    ml_type = train_info['train_info']['classifier_type']

    train_config = train_info[name]
    tfidf_config = train_config[tfidf_config_name]
    ml_config = train_config[ml_config_name]

    topic_method = tfidf_config['topic_method']

    kernel = ml_config['kernel']
    C = ml_config['C']
    gamma = ml_config['gamma']
    is_unbalanced = ml_config['is_unbalanced']
    is_prob = ml_config['is_prob']
    is_one_vs_rest = ml_config['is_one_vs_rest']
    n_jobs = ml_config['n_jobs']
    is_test = ml_config['is_test']

    class_weight = None
    if is_unbalanced:
        class_weight = 'balanced'

    test_score = None
    clf = SVC(kernel=kernel, C=C, gamma=gamma,
              class_weight=class_weight, probability=is_prob)
    if is_one_vs_rest:
        clf = OneVsRestClassifier(clf, n_jobs=n_jobs)

    if is_test:
        # split data to train and test
        X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=0.33,
                                                            shuffle=True,
                                                            random_state=42)

        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

    else:
        clf.fit(data, target)
        train_score = clf.score(data, target)

    if is_save:
        base_dir = './data/gensim/{}'.format(name)
        base_dir = os.path.join(base_dir, 'model')

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        clf_name = '{0}-{1}-{2}.pkl'.format(name, ml_type, topic_method)
        clf_path = os.path.join(base_dir, clf_name)

        with open(clf_path, 'wb') as f:
            pickle.dump(clf, f, protocol=2)

        le_name = '{0}-{1}-{2}-label_encoder.pkl'.format(
            name, ml_type, topic_method)
        le_path = os.path.join(base_dir, le_name)

        with open(le_path, 'wb') as f:
            pickle.dump(le, f, protocol=2)

    return train_score, test_score


def recover_svm_clf(train_info):

    train_common_config = train_info['train_info']

    name = train_common_config['name']

    tfidf_config_name = train_info['train_info']['tfidf_config']
    ml_type = train_info['train_info']['classifier_type']

    train_config = train_info[name]
    tfidf_config = train_config[tfidf_config_name]

    topic_method = tfidf_config['topic_method']

    base_dir = './data/gensim/{}'.format(name)

    base_dir = os.path.join(base_dir, 'model')

    clf_name = '{0}-{1}-{2}.pkl'.format(name, ml_type, topic_method)
    clf_path = os.path.join(base_dir, clf_name)

    le_name = '{0}-{1}-{2}-label_encoder.pkl'.format(
        name, ml_type, topic_method)
    le_path = os.path.join(base_dir, le_name)

    if not os.path.exists(clf_path) or not os.path.exists(le_path):
        logger.log.error('clf %s or label_encoder %s not exists' %
                         (clf_path, le_path))
        return None

    with open(clf_path, 'rb') as f:
        clf = pickle.load(f)

    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    return clf, le
