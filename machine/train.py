#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import os

import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from machine import lsi
from utils.wrapper import time_consumed
from machine import svm, logistic, xgb
from datetime import datetime


@time_consumed
def train(train_info):
    start_time = datetime.now()

    train_common_config = train_info['train_info']

    name = train_common_config['name']

    tfidf_config_name = train_info['train_info']['tfidf_config']
    ml_config_name = train_info['train_info']['classifier_config']
    ml_type = train_info['train_info']['classifier_type']

    train_config = train_info[name]
    tfidf_config = train_config[tfidf_config_name]
    ml_config = train_config[ml_config_name]

    topic_method = tfidf_config['topic_method']
    num_topics = tfidf_config['num_topics']

    data = lsi.compute_lsi_lda(train_info, recover_d=True,
                               recover_m=True, recover_tfidf=True)

    targets = [x[1] for x in data]

    if ml_type == 'xgb':
        le = LabelBinarizer()
    else:
        le = LabelEncoder()

    le.fit(targets)

    X_data = []
    Y_data = []
    for i in range(len(data)):
        x = data[i][0]
        assert np.shape(x)[1] == num_topics

        y = [targets[i]] * len(x)

        y = le.transform(y)

        X_data.append(x)
        Y_data.append(y)

    data = np.concatenate(X_data, axis=0)
    target = np.concatenate(Y_data, axis=0)

    train_score = None

    if ml_type == 'svm':
        train_score, test_score = svm.svm_clf(
            train_info, data, target, le, is_save=True)
    # TODO
    elif ml_type == 'xgb':
        pass
        # xgb.xgb_clf(data, target, le, is_save=True)
    # TODO       
    elif ml_type == 'logistic':
        pass
        # train_score = logistic.logistic_clf(data, target, le, is_save=True)
        # cross_scores = logistic.logistic_cross(data, target)

    base_dir = './data/gensim/{}'.format(name)
    base_dir = os.path.join(base_dir, 'train')

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    end_time = datetime.now()
    interval = str(end_time - start_time)

    train_date = start_time.strftime('%Y-%m-%d')
    res_name = '{0}-{1}-{2}-{3}-train.txt'.format(
        name, ml_type, topic_method, train_date)

    res_path = os.path.join(base_dir, res_name)

    with open(res_path, 'a') as f:
        f.write('time:\n')
        f.write('\tstart => %s\n' % start_time)
        f.write('\tend => %s\n' % end_time)
        f.write('\tinterval => %s\n\n' % interval)

        train_info = ['\t' + str(x) + ' => ' + str(y)
                      for x, y in train_common_config.items()]
        train_info = '\n'.join(train_info)
        f.write('train-info:\n')
        f.write(train_info)
        f.write('\n\n')

        tfidf_config = ['\t' + str(x) + ' => ' + str(y)
                        for x, y in tfidf_config.items()]
        tfidf_config = '\n'.join(tfidf_config)
        f.write('tfidf-config:\n')
        f.write(tfidf_config)
        f.write('\n\n')

        ml_config = ['\t' + str(x) + ' => ' + str(y) for x, y in ml_config.items()]
        ml_config = '\n'.join(ml_config)
        f.write('ml-config:\n')
        f.write(ml_config)
        f.write('\n\n')

        if train_score:
            f.write('score:\n')
            f.write('\ttrain_score => %s\n' % str(train_score))
            if test_score:
                f.write('\ttest_score => %s\n' % str(test_score))
            f.write('\n\n')

        f.write('*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#* \n')
        f.write('\n')
