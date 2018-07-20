import datetime
import time
from functools import wraps
from utils.logger import logger


def time_consumed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elaps = str(datetime.timedelta(seconds=end-start))
        logger.log.info('%s time consumed: %s' % (func.__name__, elaps))
        return result

    return wrapper
