from aip import AipImageClassify
from utils.cfg_parser import CfgParser
from utils.logger import logger


def get_file_content(file_path):
    with open(file_path, 'rb') as f:
        return f.read()


def main():
    log_file_path = './logs/baidu_test'
    logger.logger_init(log_file_path, stdout_level='info')

    cfg_file_path = './conf/config.ini'
    cfg_parser = CfgParser(cfg_file_path)
    baidu_info = cfg_parser.get_cfg_dict('baidu_chungu')

    APP_ID = baidu_info.get('APP_ID')
    API_KEY = baidu_info.get('API_KEY')
    SECRET_KEY = baidu_info.get('SECRET_KEY')

    client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

    image = get_file_content('./data/baidu_test/test4.png')

    res = client.logoSearch(image)
    print(res)

    # brief = "{\"name\": \"中科金财\",\"code\":\"668\"}"
    # x = client.logoAdd(image, brief)
    # print(x)

    # x = client.logoDeleteByImage(image)
    # print(x)
    


