#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import os

import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from machine import lsi
from utils.wrapper import time_consumed
from machine import svm, logistic, xgb, linear_svm
from datetime import datetime
from machine.get_project_config import get_project_config


@time_consumed
def train(project_info):
    start_time = datetime.now()

    name, dict_config, topic_method, topic_config, train_method, train_config = get_project_config(project_info)

    data = lsi.compute_lsi_lda(project_info, recover_d=True,
                               recover_m=True, recover_tfidf=True)

    targets = [x[1] for x in data]

    if train_method == 'xgb':
        le = LabelBinarizer()
    else:
        le = LabelEncoder()

    le.fit(targets)

    X_data = []
    Y_data = []
    for i in range(len(data)):
        x = data[i][0]
        assert np.shape(x)[1] == topic_config['num_topics']

        y = [targets[i]] * len(x)

        y = le.transform(y)

        X_data.append(x)
        Y_data.append(y)

    data = np.concatenate(X_data, axis=0)
    target = np.concatenate(Y_data, axis=0)

    train_score = None

    if train_method == 'svm':
        train_score, test_score = svm.clf(
            project_info, data, target, le, is_save=True)

    # TODO
    elif train_method == 'xgb':
        pass
      
    elif train_method == 'logistic':
        train_score, test_score = logistic.clf(project_info, data, target, le, is_save=True)
        # train_score = logistic.logistic_clf(data, target, le, is_save=True)
        # cross_scores = logistic.logistic_cross(data, target)
    elif train_method == 'lsvm':
        train_score, test_score = linear_svm.clf(
            project_info, data, target, le, is_save=True)

    base_dir = './data/gensim/{}'.format(name)
    base_dir = os.path.join(base_dir, 'train')

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    end_time = datetime.now()
    interval = str(end_time - start_time)

    train_date = start_time.strftime('%Y-%m-%d')
    res_name = '{0}-{1}-{2}-{3}-train.txt'.format(
        name, train_method, topic_method, train_date)

    res_path = os.path.join(base_dir, res_name)

    with open(res_path, 'a') as f:
        f.write('time:\n')
        f.write('\tstart => %s\n' % start_time)
        f.write('\tend => %s\n' % end_time)
        f.write('\tinterval => %s\n\n' % interval)

        project_info = ['\t' + str(x) + ' => ' + str(y)
                       for x, y in project_info.items() if isinstance(y, str)]
        project_info = '\n'.join(project_info)
        f.write('project-info:\n')
        f.write(project_info)
        f.write('\n\n')

        dict_config = ['\t' + str(x) + ' => ' + str(y)
                       for x, y in dict_config.items()]
        dict_config = '\n'.join(dict_config)
        f.write('dictionary-config:\n')
        f.write(dict_config)
        f.write('\n\n')

        topic_config = ['\t' + str(x) + ' => ' + str(y)
                       for x, y in topic_config.items()]
        topic_config = '\n'.join(topic_config)
        f.write('topic-config:\n')
        f.write(topic_config)
        f.write('\n\n')


        train_config = ['\t' + str(x) + ' => ' + str(y) for x, y in train_config.items()]
        train_config = '\n'.join(train_config)
        f.write('train-config:\n')
        f.write(train_config)
        f.write('\n\n')

        if train_score:
            f.write('score:\n')
            f.write('\ttrain_score => %s\n' % str(train_score))
            if test_score:
                f.write('\ttest_score => %s\n' % str(test_score))
            f.write('\n\n')

        f.write('*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#* \n')
        f.write('\n')
