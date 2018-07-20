#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
from configparser import ConfigParser

from utils.logger import logger


class CfgParser(object):

    def __init__(self, cfg_file_path):
        self.cfg_parser = None
        self.info = {}

        if not cfg_file_path or not os.path.exists(cfg_file_path):
            raise ValueError('can not find config file: %s' % cfg_file_path)

        self.cfg_parser = ConfigParser()
        self.cfg_parser.optionxform = str
        self.cfg_parser.read(cfg_file_path)

        for section in self.cfg_parser.sections():
            self.info[section] = {}
            for key, value in self.cfg_parser.items(section):
                self.info[section][key] = value

    def get_cfg_dict(self, section):

        if section not in self.cfg_parser.sections():
            logger.log.error('section %s not in config file %s' % (section, self.cfg_parser.sections()))
            raise ValueError('section %s not in config file %s' % (section, self.cfg_parser.sections()))

        return self.info[section]


class SingleCfgParser(object):

    __cfg_parser = None
    __info = {}

    @classmethod
    def init_cfg(cls, cfg_file_path):
        
        if not cfg_file_path or not os.path.exists(cfg_file_path):
            raise ValueError('can not find config file: %s' % cfg_file_path)

        cls.__cfg_parser = ConfigParser()
        cls.__cfg_parser.optionxform = str
        cls.__cfg_parser.read(cfg_file_path)

        for section in cls.__cfg_parser.sections():
            cls.__info[section] = {}
            for key, value in cls.__cfg_parser.items(section):
                cls.__info[section][key] = value

    @classmethod
    def get_cfg_dict(cls, section):

        if section not in cls.__cfg_parser.sections():
            logger.log.error('section %s not in config file %s' % (section, cls.__cfg_parser.sections()))
            raise ValueError('section %s not in config file %s' % (section, cls.__cfg_parser.sections()))

        return cls.__info[section]


def get_train_info():
    
    train_info = SingleCfgParser.get_cfg_dict('train_info')
    tfidf_config = SingleCfgParser.get_cfg_dict(train_info.get('tfidf_config'))
    ml_config = SingleCfgParser.get_cfg_dict(train_info.get('train_config'))

    return train_info, tfidf_config, ml_config