from pprint import pprint

import yaml

from machine import lsi, predict, train
# from project.keyword import keyword_filter_main
from utils.logger import logger
from utils.mysql_ctrl import MysqlCtrl
from datetime import datetime


def predict_test(config_info, train_info):
    recommend_db = MysqlCtrl(config_info['recommend_mysql_r'])
    recommend_db.connect()

    select_sql = 'SELECT news_id, content FROM t_news_corpus_latest LIMIT 10;'

    ret, news = recommend_db.TB_select(select_sql)

    tfidf_model, clf, le = predict.init_model(train_info)

    for news_id, content in news:
        pred = predict.predict(train_info, tfidf_model, clf, le, content)
        pred = le.inverse_transform(pred)

        # 
        print(pred)

    recommend_db.close()


def test():
    import matplotlib.pyplot as plt
    import glob
    import numpy as np

    # plt.style.use('ggplot')

    pattern = './data/gensim/daily/corpus/daily-train-*-*-lda.npy'

    doc_path_list = glob.glob(pattern)
    doc_path_list = sorted(doc_path_list)

    plt.figure(1)

    for j in range(70, 80):
        x = []
        y = []
        for i, doc in enumerate(doc_path_list):
            name = doc.split('-')[-3]

            day_date = datetime.strptime(name, '%Y_%m_%d')
            
            data = np.load(doc)
            
            max_index = np.argmax(data, axis=1)
            unique, count = np.unique(max_index, return_counts=True)

            x.append(day_date)
            y.append(dict(zip(unique, count)).get(j, 0))

        plt.subplot(2, 5, j%10+1)
        plt.plot(range(len(y)), y)
        plt.xlabel(j)
        plt.xticks()

    plt.show()


def main():

    log_file_path = './logs/train_daily'
    logger.logger_init(log_file_path, stdout_level='info')

    train_yaml_path = './conf/train-daily.yaml'
    config_yaml_path = './conf/config.yaml'

    with open(train_yaml_path, 'r') as f:
        project_info = yaml.load(f.read())

    with open(config_yaml_path, 'r') as f:
        config_info = yaml.load(f.read())

    try:
        # lsi.init_tfidf(project_info)

        # lsi.init_lsi_lda(project_info, recover_tfidf=True)

        # lsi.compute_lsi_lda(project_info, recover_m=False, recover_d=False, recover_tfidf=False)

        # train.train(project_info)

        # predict_test(config_info, project_info)

        # print('hello')
        test()

    except KeyboardInterrupt:

        logger.log.warn('shut down program')


if __name__ == '__main__':
    main()
