import yaml

from utils.logger import logger
from project.baidu_index.login import search_index_single, search_index_mult
import pandas as pd


def main():

    log_file_path = './logs/baidu_index'
    logger.logger_init(log_file_path, stdout_level='info')

    baidu_yaml_path = './conf/baidu-index.yaml'

    with open(baidu_yaml_path, 'r') as f:
        baidu_info = yaml.load(f.read())
    baidu_info = baidu_info['baidu_index']

    try:
        # login(baidu_info)
        columns=['公司名称', '整体日均值', '移动日均值', '整体同比', '整体环比', '移动同比', '移动环比']
        res = []

        keyword = ['云锋金融']
        x, y, browser = search_index_single(keyword, baidu_info)
        res.extend(x)
        
        keywords_list1 = ['老虎证券', '富途证券', '雪盈证券']
        x, y, browser = search_index_mult(keywords_list1, baidu_info, browser)
        res.extend(x)

        keywords_list2 = ['蓝海智投', '理财魔方', '诺亚财富', '蛋卷基金', '天天基金']
        x, y, browser = search_index_mult(keywords_list2, baidu_info, browser)
        res.extend(x)

        df = pd.DataFrame(res, columns=columns)

        df.to_excel('./data/baidu_index/baidu-index.xlsx', index=False)

        print('hello')
    except KeyboardInterrupt:
        logger.log.warn('shut down program')

if __name__ == '__main__':
    main()
