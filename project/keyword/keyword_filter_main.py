#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import pickle
import sys
from datetime import datetime, timedelta

from utils import mysql_ctrl
from utils.cfg_parser import CfgParser
from nlp_tools.keyword_filter import TrieCheck
from utils.logger import logger
from utils.wrapper import time_consumed
import shutil
import requests
from project.business_filter import predict


MARKER = [u'。', u'？', u'！']
PRE_COUNT = 3
POST_COUNT = 3
SUMMARY_LENGTH = 400

def load_keywords(file_path):
    keywords = []
    with open(file_path, 'r') as f:
        for line in f:
            if sys.version_info.major == 2:
                line = line.decode('utf-8')
            keywords.append(line.strip('\n').strip())
    return keywords


def load_news(flag=False):
    cfg_file_path = './conf/config.ini'
    cfg_parser = CfgParser(cfg_file_path)
    db_info = cfg_parser.get_cfg_dict('recommend_mysql_r')

    yff_db = mysql_ctrl.MysqlCtrl(db_info=db_info)
    ret = yff_db.connect()

    if not ret:
        logger.log.error('connect to database error, exit')
        sys.exit(-1)

    max_update_time = datetime.now() - timedelta(days=2)
    file_path = './data/keyword/max_update_time.pkl'
    if not flag and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            max_update_time = pickle.load(f)

    select_sql1 = 'SELECT news_id, original_url, media, info_publ_date, info_title, content, update_time, src_type '\
                  'FROM t_news_corpus_latest '\
                  'WHERE update_time > "%s" '\
                  'ORDER BY info_publ_date;' % max_update_time

    select_sql2 = 'SELECT tag_id, tag_name '\
                  'FROM t_news_tag '\
                  'WHERE tag_id LIKE "01%";'

    ret1, news = yff_db.TB_select(select_sql1)
    ret2, news_tag = yff_db.TB_select(select_sql2)

    news_tag = dict(news_tag)

    if ret1 and ret2 and len(news) > 0:
        max_update_time = max([item[-1] for item in news])
        with open(file_path, 'wb') as f:
            pickle.dump(max_update_time, f, True)

        logger.log.info(
            'loaded %d news and news_tag successfully!' % len(news))
    elif len(news) == 0:
        logger.log.info(
            'no news to process')
    else:
        logger.log.error('loaded news failed!')
        sys.exit(-1)

    yff_db.close()

    return news, news_tag


def get_checker(type_filter):
    assert type_filter in ['company', 'stockholder', 'business', 'adversary']
    checker = TrieCheck()
    keywords_file_path = './data/keyword/%s_keyword.txt' % type_filter
    keywords = load_keywords(keywords_file_path)

    for word in keywords:
        checker.add_keyword(word)

    return checker


def save_summary(index, news_id, url, media, publ_date, 
                 title, keyword, content, f, title_summary=False):

    if not title_summary:
        i, j = index, index
        start_index = 0
        end_index = len(content) - 1

        pre_count, post_count = 0, 0
        while i > 0 or j < len(content) - 1:
            if i > 0 and pre_count < PRE_COUNT:
                i -= 1
                if content[i] in MARKER:
                    start_index = i + 1
                    pre_count += 1

            if j < len(content) - 1 and post_count < POST_COUNT:
                j += 1
                if content[j] in MARKER:
                    end_index = j
                    post_count += 1
            
            if (i == 0 or pre_count >= PRE_COUNT) and (post_count >= POST_COUNT or j == len(content) - 1):
                break
    else:
        k = 0
        post_count = 0
        start_index = 0
        end_index = 0
        while k < len(content) - 1 and post_count < PRE_COUNT + POST_COUNT:
            k += 1
            if content[k] in MARKER:
                end_index = k
                post_count += 1

    summary = content[start_index:end_index+1]

    if sys.version_info.major == 2:
        f.write(u'新闻ID: %s\n'.encode('utf8') % news_id)
        f.write(u'新闻链接: %s\n'.encode('utf8') % url.encode('utf8'))
        f.write(u'新闻来源: %s\n'.encode('utf8') % media.encode('utf8'))
        f.write(u'发布日期: %s\n'.encode('utf8') % publ_date.encode('utf8'))
        f.write(u'新闻标题: %s\n'.encode('utf8') % title.encode('utf8'))
        f.write(u'关键词: %s\n'.encode('utf8') % keyword.encode('utf8'))
        f.write(u'新闻摘要: %s\n'.encode('utf8') % summary.encode('utf8'))
        f.write('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')
    else:
        f.write(u'新闻ID: %s\n' % str(news_id))
        f.write(u'新闻链接: %s\n' % url)
        f.write(u'新闻来源: %s\n' % media)
        f.write(u'发布日期: %s\n' % publ_date)
        f.write(u'新闻标题: %s\n' % title)
        f.write(u'关键词: %s\n' % keyword)
        f.write(u'新闻摘要: %s\n' % summary)
        f.write('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')


def get_summary_file_path(type_filter):
    assert type_filter in ['company', 'stockholder', 'business', 'adversary']
    summary_file_name = type_filter + '-' + datetime.now().strftime('%Y-%m-%d') + '.txt'
    summary_dir_name = 'summary' + '-' + datetime.now().strftime('%Y-%m-%d')
    summary_dir_name = os.path.join('./data/keyword/summary', summary_dir_name)
    if not os.path.exists(summary_dir_name):
        os.mkdir(summary_dir_name)
    summary_file_path = os.path.join(summary_dir_name, summary_file_name)
    
    return summary_file_path


def company_filter(news, news_dict):
    type_filter = 'company'
    checker = get_checker(type_filter)

    summary_file_name = get_summary_file_path(type_filter)
    f = open(summary_file_name, 'w')

    for item in news:
        news_id = item[0]
        url = item[1]
        media = item[2]
        media = news_dict.get(media)
        publ_date = item[3]
        title = item[4]
        content = item[5]

        src_type = item[-1]

        if src_type == 'shuwen':
            media = u'数文'

        res = checker.get_keyword(content)
        publ_date = publ_date.strftime('%Y-%m-%d')

        if res:
            processed = set()

            for index, keyword in res:
                if keyword in processed:
                    continue
                processed.add(keyword)

                keyword_count = [(x, y) for x, y in res if y == keyword]
                if len(keyword_count) > 1:
                    index = keyword_count[len(keyword_count) // 2][0]
                save_summary(index, news_id, url, media, publ_date, title, keyword, content, f)
                
    f.close()


def stockholder_filter(news, news_dict):
    type_filter = 'stockholder'
    checker = get_checker(type_filter)

    summary_file_name = get_summary_file_path(type_filter)
    f = open(summary_file_name, 'w')

    for item in news:
        news_id = item[0]
        url = item[1]
        media = item[2]
        media = news_dict.get(media)
        publ_date = item[3]
        title = item[4]
        content = item[5]

        res = checker.get_keyword(content)
        publ_date = publ_date.strftime('%Y-%m-%d')

        src_type = item[-1]

        if src_type == 'shuwen':
            media = u'数文'

        if res:
            processed = set()

            for index, keyword in res:
                if keyword in processed:
                    continue
                processed.add(keyword)
  
                keyword_count = [(x, y) for x, y in res if y == keyword]
                if len(keyword_count) > 1:
                    index = keyword_count[len(keyword_count) // 2][0]
                save_summary(index, news_id, url, media, publ_date, title, keyword, content, f)

    f.close()


def business_filter(news, news_dict):
    type_filter = 'business'
    checker = get_checker(type_filter)

    summary_file_name = get_summary_file_path(type_filter)
    f = open(summary_file_name, 'w')

    # 采集不相关的数据作为训练数据
    temp = []

    tfidf_model, clf, le = predict.init_model()

    for item in news:
        news_id = item[0]
        url = item[1]
        media = item[2]
        media = news_dict.get(media)
        publ_date = item[3]
        title = item[4]
        content = item[5]

        publ_date = publ_date.strftime('%Y-%m-%d')

        src_type = item[-1]

        if src_type == 'shuwen':
            media = u'数文'

        res_content = checker.get_keyword(content)
        res_title = checker.get_keyword(title)

        pred = predict.predict(tfidf_model, clf, le, content)
        
        if sys.version_info.major == 2:
            prob_txt = u'相关概率: %f, 不相关概率: %f' % (pred[0, 0], pred[0, 1])
        else:
            prob_txt = '相关概率: %f, 不相关概率: %f' % (pred[0, 0], pred[0, 1])
        # print(type(publ_date), type(prob_txt))
        publ_date = publ_date + ':      ' + prob_txt

        if len(res_title) == 0 and len(res_content) == 0:
            temp.append((title, content, publ_date, 1))
            continue

        elif len(res_title) == 0 and len(res_content) > 0:
            processed = set()

            for index, keyword in res_content:
                if keyword in processed:
                    continue
                processed.add(keyword)
  
                keyword_count = [(x, y) for x, y in res_content if y == keyword]
                if len(keyword_count) > 1:
                    index = keyword_count[len(keyword_count) // 2][0]
                save_summary(index, news_id, url, media, publ_date, title, keyword, content, f)

        elif len(res_title) > 0 and len(res_content) == 0:
            index, keyword = res_title[0][0], res_title[0][1]
            save_summary(index, news_id, url, media, publ_date, title, keyword, content, f, title_summary=True)

        elif len(res_title) > 0 and len(res_content) > 0:
            keyword_content = [x[1] for x in res_content]
            keyword_content_title = [x for _, x in res_title if x in keyword_content]

            if len(keyword_content_title) == 0:
                index, keyword = res_title[0][0], res_title[0][1]
                save_summary(index, news_id, url, media, publ_date, title, keyword, content, f, title_summary=True)
            else:
                keyword = keyword_content_title[0]
                keyword_count = [(x, y) for x, y in res_content if y == keyword]
           
                index = keyword_count[len(keyword_count) // 2][0]
                save_summary(index, news_id, url, media, publ_date, title, keyword, content, f)
    f.close()

    with open('./data/keyword/temp.pkl', 'wb') as f:
        pickle.dump(temp, f, True)


def adversary_keyword_filter(index, keyword, content):
    if content[index-5:index].endswith(u'来源：'):
        return True
    elif content[index+len(keyword):index+8].startswith(u'·专栏作者'):
        return True
    else:
        return False


def adversary_filter(news, news_dict):
    type_filter = 'adversary'
    checker = get_checker(type_filter)

    summary_file_name = get_summary_file_path(type_filter)
    f = open(summary_file_name, 'w')

    for item in news:
        news_id = item[0]
        url = item[1]
        media = item[2]
        media = news_dict.get(media)
        publ_date = item[3]
        title = item[4]
        content = item[5]

        res = checker.get_keyword(content)
        publ_date = publ_date.strftime('%Y-%m-%d')

        src_type = item[-1]

        if src_type == 'shuwen':
            media = u'数文'

        if res:
            processed = set()

            for index, keyword in res:
                if keyword in processed:
                    continue
  
                if adversary_keyword_filter(index, keyword, content):
                    continue
                
                processed.add(keyword)

                save_summary(index, news_id, url, media, publ_date, title, keyword, content, f)

    f.close()


def delete_empty():
    summary_dir_name = 'summary' + '-' + datetime.now().strftime('%Y-%m-%d')
    summary_dir_name = os.path.join('./data/keyword/summary', summary_dir_name)

    summary_files = os.listdir(summary_dir_name)
    
    if len(summary_dir_name) > 0:
        for file in summary_files:
            file_path = os.path.join(summary_dir_name, file)
            with open(file_path, 'r') as f:
                content = f.read()
                if not content.strip('\n').strip():
                    os.remove(file_path)
    
    summary_files = os.listdir(summary_dir_name)
    if len(summary_files) == 0:
        return False
    else:
        return True


def dingtalk(message_url):
    cfg_file_path = './conf/config.ini'
    cfg_parser = CfgParser(cfg_file_path)
    db_info = cfg_parser.get_cfg_dict('dingtalk')
    robot_token = db_info.get('robot_token')

    url = 'https://oapi.dingtalk.com/robot/send?access_token=' + robot_token

    json1 = {
        "msgtype": "text",
        "text": {
            "content": "新闻更新-%s" % datetime.now().strftime('%Y-%m-%d')
        },
        "at": {
            "atMobiles": [
                "13502066772"
            ],
            "isAtAll": 'false'
        }
    }

    json = {
            'msgtype': 'link',
            'link': {
                'title': '新闻更新-%s' % datetime.now().strftime('%Y-%m-%d'),
                'text': '新闻更新-%s' % datetime.now().strftime('%Y-%m-%d'),
                'messageUrl': message_url
            }
        }

    requests.post(url, json=json1)
    requests.post(url, json=json)


def make_summary_archive():
    ret = delete_empty()
    url = 'http://10.9.19.251:9898/'
    if ret:
        summary_tar_name = 'summary' + '-' + datetime.now().strftime('%Y-%m-%d')
        summary_tar_path = os.path.join('./data/keyword/summary/archive', summary_tar_name)
        root_dir = os.path.join('./data/keyword/summary', summary_tar_name)
        shutil.make_archive(summary_tar_path, format='zip',
                        root_dir=root_dir)
        url += summary_tar_name + '.zip'
        dingtalk(url)
        return True
    else:
        return False


def main():

    log_file_path = './logs/keyword_filter'
    logger.logger_init(log_file_path, fileout_level='info')

    news, news_dict = load_news(flag=True)

    company_filter(news, news_dict)

    stockholder_filter(news, news_dict)
    business_filter(news, news_dict)
    adversary_filter(news, news_dict)

    make_summary_archive()


if __name__ == '__main__':
    main()
