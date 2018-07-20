#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import glob
import os
import sys
from pprint import pprint

import gensim
import numpy as np
from gensim import corpora, models
from six import iteritems

from utils.logger import logger


class MyDoc(object):
    def __init__(self, doc_path_list):
        self.doc_path_list = doc_path_list

    def __iter__(self):
        if isinstance(self.doc_path_list, list):
            for doc_path in self.doc_path_list:
                if not os.path.exists(doc_path):
                    logger.log.error('doc file %s not exists' % doc_path)
                else:
                    logger.log.info('loading doc file %s' % doc_path)
                    for line in open(doc_path):
                        yield line.split(',')
        else:
            doc_path = self.doc_path_list
            if not os.path.exists(doc_path):
                logger.log.error('doc file %s not exists' % doc_path)
            else:
                logger.log.info('loading doc file %s' % doc_path)
                for line in open(doc_path):
                    yield line.split(',')


class MyCorpus(object):
    def __init__(self, my_doc, dictionary):
        self.my_doc = my_doc
        self.dictionary = dictionary

    def __iter__(self):
        for doc in self.my_doc:
            yield self.dictionary.doc2bow(doc)


class GensimTfidf(object):

    def __init__(self, name='test'):

        self.dictionary = corpora.Dictionary()
        self.tfidf_model = None
        self.lsi_model = None
        self.lda_model = None

        self.name = name

        self.base_dir = './data/gensim/{0}'.format(self.name)

        self.dictionay_path = './data/gensim/{0}/dictionary/{0}.dict'.format(
            self.name)
        self.tfidf_model_path = './data/gensim/{0}/model/{0}-tfidf.model'.format(
            self.name)
        self.lsi_model_path = './data/gensim/{0}/model/{0}-lsi.model'.format(
            self.name)
        self.lda_model_path = './data/gensim/{0}/model/{0}-lda.model'.format(
            self.name)

    def load_dictionary(self):
        if not os.path.exists(self.dictionay_path):
            logger.log.error(
                'saved dictionary file %s not exists!' % self.dictionay_path)
        else:
            logger.log.info('loading saved dictionary ...')
            self.dictionary = corpora.Dictionary.load(self.dictionay_path)

    def save_dictionary(self):
        dictionary_dir = os.path.join(self.base_dir, 'dictionary')
        if not os.path.exists(dictionary_dir):
            os.makedirs(dictionary_dir)
        logger.log.info('saving dictionary ...')
        self.dictionary.save(self.dictionay_path)

    def load_tfidf_model(self):
        if not os.path.exists(self.tfidf_model_path):
            logger.log.error('saved tfidf model %s not exists!' %
                             self.tfidf_model_path)
        else:
            logger.log.info('loading saved tfidf model ...')
            self.tfidf_model = models.TfidfModel.load(self.tfidf_model_path)

    def save_tfidf_model(self):
        model_dir = os.path.join(self.base_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        logger.log.info('saving tfidf model ...')
        self.tfidf_model.save(self.tfidf_model_path)

    def load_lsi_model(self):
        if not os.path.exists(self.lsi_model_path):
            logger.log.error('saved lsi model %s not exists!' %
                             self.lsi_model_path)
        else:
            logger.log.info('loading saved lsi model ...')
            self.lsi_model = models.LsiModel.load(self.lsi_model_path)

    def save_lsi_model(self):
        model_dir = os.path.join(self.base_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        logger.log.info('saving lsi model ...')
        self.lsi_model.save(self.lsi_model_path)

    def load_lda_model(self):
        if not os.path.exists(self.lda_model_path):
            logger.log.error('saved lda model %s not exists!' %
                             self.lda_model_path)
        else:
            logger.log.info('loading saved lsi model ...')
            self.lda_model = models.LdaModel.load(self.lda_model_path)

    def save_lda_model(self):
        model_dir = os.path.join(self.base_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        logger.log.info('saving lda model ...')
        self.lda_model.save(self.lda_model_path)

    def add_document_from_file(self, my_doc, is_save=False, count_threshold=2):
        if not isinstance(my_doc, MyDoc):
            logger.log.error('please provide instance of MyDoc')
        else:
            self.dictionary.add_documents(line for line in my_doc)
            rare_ids = []
            for tokenid, doc_freq in iteritems(self.dictionary.dfs):
                if doc_freq <= count_threshold:
                    rare_ids.append(tokenid)
            self.dictionary.filter_tokens(rare_ids)
            if is_save:
                self.save_dictionary()

    def add_document_from_var(self, doc_list):
        self.dictionary.add_documents(doc_list)

    def init_tfidf_from_file(self, my_doc, is_save=False):

        if not isinstance(my_doc, MyDoc):
            logger.log.error('please provide instance of MyDoc!')
        else:
            my_corpus = MyCorpus(my_doc, self.dictionary)
            self.tfidf_model = models.TfidfModel(my_corpus)

            if is_save:
                self.save_tfidf_model()

    def init_tfidf_from_corpus(self, corpus):
        self.tfidf_model = models.TfidfModel(corpus)

    def compute_tfidf(self, my_doc, is_file=True, is_save=False,
                      corpus_name='test'):

        if not is_file:
            if not isinstance(my_doc, list):
                raise TypeError('my doc should be a list')
            doc = [self.dictionary.doc2bow(x) for x in my_doc]
            corpus_tfidf = self.tfidf_model[doc]
            return corpus_tfidf

        if not isinstance(my_doc, MyDoc):
            logger.log.error('please provide instance of MyDoc!')
            return []
        elif not self.tfidf_model:
            logger.log.error('please initial tfidf model first!')
            return []
        else:
            my_corpus = MyCorpus(my_doc, self.dictionary)
            corpus_tfidf = self.tfidf_model[my_corpus]

            if is_save:
                corpus_dir = os.path.join(self.base_dir, 'corpus')
                if not os.path.exists(corpus_dir):
                    os.makedirs(corpus_dir)
                corpus_path = os.path.join(
                    corpus_dir, '{}-tfidf.mm'.format(corpus_name))
                logger.log.info('save corpus-tfidf to %s' % corpus_path)
                corpora.MmCorpus.serialize(corpus_path, corpus_tfidf)

            return corpus_tfidf

    def init_lsi_model(self, corpus_tfidf, num_topics=2, is_save=False):
        self.lsi_model = models.LsiModel(
            corpus_tfidf, id2word=self.dictionary, num_topics=num_topics)
        if is_save:
            self.save_lsi_model()

    def init_lda_model(self, corpus_tfidf, num_topics=2, is_save=False):
        self.lda_model = models.LdaModel(
            corpus_tfidf, id2word=self.dictionary, num_topics=num_topics)
        if is_save:
            self.save_lda_model()

    def update_lsi_model(self, corpus_tfidf, num_topics=2, is_save=False):
        self.lsi_model.add_documents(corpus_tfidf)
        if is_save:
            self.save_lsi_model()

    def compute_lsi(self, corpus_tfidf, num_topics,
                    is_dense=False, is_save=False, corpus_name='test',
                    is_npy_save=False):
        corpus_lsi = self.lsi_model[corpus_tfidf]
        if is_save:
            corpus_dir = os.path.join(self.base_dir, 'corpus')
            if not os.path.exists(corpus_dir):
                os.makedirs(corpus_dir)
            corpus_path = os.path.join(
                corpus_dir, '{}-lsi.mm'.format(corpus_name))
            logger.log.info('save corpus-lsi to %s' % corpus_path)
            corpora.MmCorpus.serialize(corpus_path, corpus_lsi)

        if is_dense:
            corpus_lsi = gensim.matutils.corpus2dense(
                corpus_lsi, num_terms=num_topics)
            corpus_lsi = corpus_lsi.T

            if is_npy_save:
                corpus_dir = os.path.join(self.base_dir, 'corpus')
                if not os.path.exists(corpus_dir):
                    os.makedirs(corpus_dir)
                corpus_path = os.path.join(
                    corpus_dir, '{}-lsi.npy'.format(corpus_name))
                logger.log.info('save corpus-lsi-npy to %s' % corpus_path)
                np.save(corpus_path, corpus_lsi)

        return corpus_lsi

    def compute_lda(self, corpus_tfidf, num_topics,
                    is_dense=False, is_save=False, corpus_name='test',
                    is_npy_save=False):
        corpus_lda = self.lda_model[corpus_tfidf]
        if is_save:
            corpus_dir = os.path.join(self.base_dir, 'corpus')
            if not os.path.exists(corpus_dir):
                os.makedirs(corpus_dir)
            corpus_path = os.path.join(
                corpus_dir, '{}-lda.mm'.format(corpus_name))
            logger.log.info('save corpus-lda to %s' % corpus_path)
            corpora.MmCorpus.serialize(corpus_path, corpus_lda)

        if is_dense:
            corpus_lda = gensim.matutils.corpus2dense(
                corpus_lda, num_terms=num_topics)
            corpus_lda = corpus_lda.T

            if is_npy_save:
                corpus_dir = os.path.join(self.base_dir, 'corpus')
                if not os.path.exists(corpus_dir):
                    os.makedirs(corpus_dir)
                corpus_path = os.path.join(
                    corpus_dir, '{}-lda.npy'.format(corpus_name))
                logger.log.info('save corpus-lda-npy to %s' % corpus_path)
                np.save(corpus_path, corpus_lda)

        return corpus_lda

    def load_tfidf_corpus(self, corpus_name):
        corpus_dir = os.path.join(self.base_dir, 'corpus')
        corpus_path = os.path.join(
            corpus_dir, '{}-tfidf.mm'.format(corpus_name))
        logger.log.info('loading corpus-tfidf from %s' % corpus_path)
        corpus_tfidf = corpora.MmCorpus(corpus_path)
        return corpus_tfidf

    def load_lsi_corpus(self, corpus_name, is_npy=False):
        corpus_dir = os.path.join(self.base_dir, 'corpus')
        corpus_path = os.path.join(
            corpus_dir, '{}-lsi.mm'.format(corpus_name))

        if not is_npy:
            logger.log.info('loading corpus-lsi from %s' % corpus_path)
            corpus_lsi = corpora.MmCorpus(corpus_path)
        else:
            corpus_path = os.path.join(
                corpus_dir, '{}-lsi.npy'.format(corpus_name))
            logger.log.info('loading corpus-lsi from %s' % corpus_path)
            corpus_lsi = np.load(corpus_path)
        return corpus_lsi

    def load_lda_corpus(self, corpus_name, is_npy=False):
        corpus_dir = os.path.join(self.base_dir, 'corpus')
        corpus_path = os.path.join(
            corpus_dir, '{}-lda.mm'.format(corpus_name))

        if not is_npy:
            logger.log.info('loading corpus-lda from %s' % corpus_path)
            corpus_lda = corpora.MmCorpus(corpus_path)
        else:
            corpus_path = os.path.join(
                corpus_dir, '{}-lda.npy'.format(corpus_name))
            logger.log.info('loading corpus-lda from %s' % corpus_path)
            corpus_lda = np.load(corpus_path)
        return corpus_lda
