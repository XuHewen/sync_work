#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from utils.logger import logger
from machine.get_project_config import get_project_config


def clf(project_info, data, target, le, is_save=True):

    name, _, topic_method, _, train_method, train_config = get_project_config(project_info)

    is_test = train_config['is_test']

    is_one_vs_rest = train_config['is_one_vs_rest']

    is_unbalanced = train_config['is_unbalanced']

    n_jobs = train_config['n_jobs']

    params = train_config['params']

    class_weight = None
    if is_unbalanced:
        class_weight = 'balanced'

    test_score = None
    clf = SVC(class_weight=class_weight, **params)

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

        clf_name = '{0}-{1}-{2}.pkl'.format(name, train_method, topic_method)
        clf_path = os.path.join(base_dir, clf_name)

        with open(clf_path, 'wb') as f:
            pickle.dump(clf, f, protocol=2)

        le_name = '{0}-{1}-{2}-label_encoder.pkl'.format(
            name, train_method, topic_method)
        le_path = os.path.join(base_dir, le_name)

        with open(le_path, 'wb') as f:
            pickle.dump(le, f, protocol=2)

    return train_score, test_score


def recover_clf(project_info):

    name, _, topic_method, _, train_method, _ = get_project_config(project_info)

    base_dir = './data/gensim/{}'.format(name)

    base_dir = os.path.join(base_dir, 'model')

    clf_name = '{0}-{1}-{2}.pkl'.format(name, train_method, topic_method)
    clf_path = os.path.join(base_dir, clf_name)

    le_name = '{0}-{1}-{2}-label_encoder.pkl'.format(
        name, train_method, topic_method)
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
