from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


from utils.logger import logger
from utils.mysql_ctrl import MysqlCtrl
import yaml
from datetime import datetime, timedelta
from machine import predict
import os
import requests
import shutil


def load_news(recommend_db-*):
    select_sql = 'SELECT news_id, title, url, publ_date, summary, keyword, keyword_type, source '\
                 'FROM t_baidu_search '\
                 'WHERE is_processed = 0;'

    _, news = recommend_db.TB_select(select_sql)

    return news


def update_processed_news(recommend_db, news_list):
    update_sql = 'UPDATE t_baidu_search '\
                 'SET is_processed = 1 '\
                 'WHERE news_id in (%s);'

    news_id_list = [str(news[0]) for news in news_list]
    update_sql = update_sql % ', '.join(news_id_list)

    ret = recommend_db.TB_update(update_sql)
    
    return ret


def update_accepted_news(recommend_db, news_list):
    update_sql = 'UPDATE t_baidu_search '\
                 'SET is_accepted = 1 '\
                 'WHERE news_id in (%s);'

    news_id_list = [str(news[0]) for news in news_list]
    update_sql = update_sql % ', '.join(news_id_list)

    ret = recommend_db.TB_update(update_sql)

    return ret


def load_source(source_file_path):
    source = set()

    with open(source_file_path, 'r') as f:
        for line in f:
            source.add(line.strip('\n').strip())

    return source


def save_news(news_list):
    today = datetime.now().strftime('%Y-%m-%d')
    news_dir = os.path.join('./data/baidu_news', 'news-{}'.format(today))
    if not os.path.exists(news_dir):
        os.mkdir(news_dir)
    
    filetypes = ['股东', '公司', '竞争对手', '业务']
    filetypes_dict = {'stockholder': '股东', 'company': '公司', 'adversary': '竞争对手', 'business': '业务'}
    filenames = ['news-{0}-{1}.txt'.format(x, today) for x in filetypes]
    file_paths = [os.path.join(news_dir, x) for x in filenames]

    f = {}
    for i, file_path in enumerate(file_paths):
        f[filetypes[i]] = open(file_path, 'w')

    for news_id, title, url, source, keyword_type, summary in news_list:
        filetype = filetypes_dict[keyword_type]
        temp_f = f[filetype]
        temp_f.write(title + '\n')
        temp_f.write(url + '\n')
        temp_f.write(source + '\n')
        temp_f.write(summary + '\n')
        temp_f.write('\n')


    for i, file_path in enumerate(file_paths):
        f[filetypes[i]].close()


def delete_empty():
    news_dir_name = 'news' + '-' + datetime.now().strftime('%Y-%m-%d')
    news_dir_path = os.path.join('./data/baidu_news', news_dir_name)

    summary_files = os.listdir(news_dir_path)
    
    if len(summary_files) > 0:
        for file in summary_files:
            file_path = os.path.join(news_dir_path, file)
            with open(file_path, 'r') as f:
                content = f.read()
                if not content.strip('\n').strip():
                    os.remove(file_path)
    
    summary_files = os.listdir(news_dir_path)

    if len(summary_files) == 0:
        return False
    else:
        return True


def dingtalk(message_url, robot_token):
    

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
        summary_tar_name = 'news' + '-' + datetime.now().strftime('%Y-%m-%d')
        summary_tar_path = os.path.join('./data/baidu_news/archive', summary_tar_name)
    
        root_dir = os.path.join('./data/baidu_news', summary_tar_name)

        shutil.make_archive(summary_tar_path, format='zip',
                            root_dir=root_dir)
        url += summary_tar_name + '.zip'

        return url
    else:
        return None


def detect():
    logger.log.info('starting filter news ... ')

    cfg_file_path = './conf/config.yaml'
    project_file_path = './conf/train-business.yaml'
    with open(cfg_file_path, 'r') as f:
        cfg_info = yaml.load(f.read())
    with open(project_file_path, 'r') as f:
        project_info = yaml.load(f.read())

    dingtalk_token = cfg_info['dingtalk']['robot_token']

    recommend_db_info = cfg_info['recommend_mysql_r']
    recommend_db = MysqlCtrl(db_info=recommend_db_info)
    recommend_db.connect()

    tfidf_model, clf, le = predict.init_model(project_info)

    news_list = load_news(recommend_db)

   
    source_file_path = './data/baidu_news/source.csv'
    source_list = load_source(source_file_path)

    res = []
    if len(news_list) > 0:
        for news_id, title, url, publ_date, summary, keyword, keyword_type, source in news_list:
            today = datetime.now()
            if source not in source_list:
                continue
            
            if today - timedelta(days=7) > publ_date:
                continue

            if keyword not in title and keyword not in summary:
                continue

            keyword_t = keyword + ':'
            if keyword_type == 'adversary' and (keyword_t in title or keyword_t in summary):
                continue

            if keyword_type == 'adversary' and (title.startswith(keyword) or summary.startswith(keyword)):
                continue

            if keyword_type == 'business':
                pred = predict.predict(project_info, tfidf_model, clf, le, summary)
                if pred[0] == 1:
                    continue

            res.append((news_id, title, url, source, keyword_type, summary))

        save_news(res)
        url = make_summary_archive()

        if url:
            dingtalk(url, dingtalk_token)

        update_processed_news(recommend_db, news_list)
        update_accepted_news(recommend_db, res)

    recommend_db.close()