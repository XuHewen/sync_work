import sys
from utils.logger import logger
from project.baidu_news import news_detect


def main():

    log_file_path = './logs/baidu_news'
    logger.logger_init(log_file_path, stdout_level='info')

    news_detect.detect()

if __name__ == '__main__':
    main()
