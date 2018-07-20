#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import os
import sys
import pickle

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.cfg_parser import get_train_info
from utils.logger import logger
import xgboost as xgb


def xgb_clf(data, target, le, is_save=True):

    train_info, tfidf_config, ml_config = get_train_info()

    # setting parameters
    name = train_info.get('name')
    ml_type = train_info.get('train_type')

    topic_method = tfidf_config.get('topic_method')

    max_depth = int(ml_config.get('max_depth'))
    eta = int(ml_config.get('eta'))
    silent = int(ml_config.get('silent'))
    objective = ml_config.get('objective')
    nthread = int(ml_config.get('nthread'))
    eval_metric = ml_config.get('eval_metric')
    num_round = int(ml_config.get('num_round'))

    params = {'max_depth': max_depth,
              'eta': eta,
              'silent': silent,
              'objective': objective,
              'nthread': nthread,
              'eval_metric': eval_metric}

    # split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.33,
                                                        shuffle=True,
                                                        random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    # train
    bst = xgb.train(params, dtrain, num_round, evallist)

    if is_save:
        base_dir = './data/gensim/{}'.format(name)
        base_dir = os.path.join(base_dir, 'model')

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        clf_name = '{0}-{1}-{2}.pkl'.format(name, ml_type, topic_method)
        clf_path = os.path.join(base_dir, clf_name)

        bst.save_model(clf_path)

        le_name = '{0}-{1}-{2}-label_encoder.pkl'.format(
            name, ml_type, topic_method)
        le_path = os.path.join(base_dir, le_name)

        with open(le_path, 'wb') as f:
            pickle.dump(le, f)


# def recover_svm_clf():

#     train_info, tfidf_config, ml_config = get_train_info()
#     name = train_info.get('name')

#     ml_type = train_info.get('train_type')
#     topic_method = tfidf_config.get('topic_method')

#     base_dir = './data/gensim/{}'.format(name)
#     base_dir = os.path.join(base_dir, 'model')

#     clf_name = '{0}-{1}-{2}.pkl'.format(name, ml_type, topic_method)
#     clf_path = os.path.join(base_dir, clf_name)

#     le_name = '{0}-{1}-{2}-label_encoder.pkl'.format(name, ml_type, topic_method)
#     le_path = os.path.join(base_dir, le_name)

#     if not os.path.exists(clf_path) or not os.path.exists(le_path):
#         logger.log.error('clf %s or label_encoder %s not exists' % (clf_path, le_path))
#         return None

#     with open(clf_path, 'rb') as f:
#         clf = pickle.load(f)

#     with open(le_path, 'rb') as f:
#         le = pickle.load(f)

#     return clf, le


# def svm_cross(data, target):

#     train_info, tfidf_config, ml_config = get_train_info()

#     kernel = str(ml_config.get('kernel'))
#     C = float(ml_config.get('C'))
#     gamma = float(ml_config.get('gamma'))
#     is_unbalanced = bool(int(ml_config.get('is_unbalanced')))
#     prob = int(ml_config.get('prob'))
#     prob = True if prob else False
#     is_one_vs_rest = int(ml_config.get('is_one_vs_rest'))
#     n_jobs = int(ml_config.get('n_jobs'))

#     class_weight = None
#     if is_unbalanced:
#         class_weight = 'balanced'

#     clf = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight, probability=prob)
#     if is_one_vs_rest:
#         clf = OneVsRestClassifier(clf, n_jobs=n_jobs)
#     scores = cross_val_score(clf, data, target, cv=10)

#     return scores
