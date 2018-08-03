#!/usr/bin/env python
#-*- coding: utf-8 -*-
import MySQLdb
import time

from utils.wrapper import time_consumed
from utils.logger import logger


class MysqlCtrl(object):

    def __init__(self, db_info=None):
        self.db_info = db_info
        self.conn = None
        self.db = None

    def reconnect(self):
        ret = self.close()
        ret = self.connect()
        return ret

    def connect(self, db_info=None):
        if db_info:
            self.db_info = db_info

        if not self.db_info:
            return False

        server = self.db_info.get('server')
        user = self.db_info.get('user')
        passwd = self.db_info.get('passwd')
        database = self.db_info.get('database')
        port = int(self.db_info.get(
            'port')) if 'port' in self.db_info else 3306

        try:
            self.conn = MySQLdb.connect(server, user,
                                        passwd, database,
                                        port, charset='utf8')
            self.db = self.conn.cursor()
            
        except Exception as e:
            logger.log.error('connect db erro, %s: %s' % (self.db_info, e))
            return False

        return True

    def TB_desc(self, table_name):
        sql = "DESC %s;" % table_name
        try:
            self.db.execute(sql)
            desc = self.db.fetchall()
        except Exception as e:
            logger.log.error('desc table error, %s: %s' % (self.db_info, e))
            return None
        return desc

    def TB_exist(self, table_name):
        sql = 'SHOW TABLES LIKE "%s";' % table_name
        try:
            self.db.execute(sql)
            items = self.db.fetchall()
        except Exception as e:
            logger.log.error('check exist error, %s: %s' % (self.db_info, e))
            items = None
            return False
        
        return len(items) != 0

    def TB_create(self, create_sql):
        logger.log.info('create table: %s' % create_sql)
        try:
            self.db.execute(create_sql)
            self.conn.commit()
        except Exception as e:
            logger.log.error('create table error, %s: %s' % (self.db_info, e))
            return False
        return True

    def TB_drop(self, table_name):
        if not self.TB_exist(table_name):
            logger.log.warn('drop table: table %s not exist' % table_name)
        
        drop_sql = 'DROP TABLE %s;' % table_name
        logger.log.info('drop table: %s' % table_name)

        try:
            self.db.execute(drop_sql)
            self.conn.commit()
        except Exception as e:
            logger.log.error('drop table error, %s: %s' % (self.db_info, e))
            return False
        return True

    def TB_rename(self, old_table_name, new_table_name, drop_if_exist=False):
        if not self.TB_exist(old_table_name):
            logger.log.warn('rename table: renamed table %s not exist' % old_table_name)
            return False

        if self.TB_exist(new_table_name):
            if drop_if_exist:
                self.TB_drop(new_table_name)
            else:
                logger.log.warn('rename table: table of rename %s exist ')
                return False

        rename_sql = "RENAME TABLE %s TO %s;" % (old_table_name, new_table_name)
        logger.log.info('rename table: %s' % rename_sql)
        try:
            self.db.execute(rename_sql)
            self.conn.commit()
        except Exception as e:
            logger.log.error('rename table error, %s: %s' % (self.db_info, e))
            return False
        return True

    def TB_insert(self, insert_sql, data_list, step_size=1000):
        logger.log.info('insert data: %s' % insert_sql)
        return self.execute_many(insert_sql, data_list, step_size)

    def execute_many(self, sql, data_list, step_size):

        step_num = len(data_list) // step_size
        if (len(data_list) % step_size) != 0:
            step_num = step_num + 1

        for i in range(step_num):
            sub_list = data_list[step_size * i: step_size * (i + 1)]
            try :
                self.db.executemany(sql, sub_list)
                self.conn.commit()
            except Exception as e:
                logger.log.error('execute_many error, %s: %s' % (self.db_info, e))
                return False
        return True

    @time_consumed
    def TB_select(self, select_sql):
        "select data"
        logger.log.info('select data: %s' % select_sql)
        try:
            self.db.execute(select_sql)
            data = self.db.fetchall()
        except Exception as e:
            logger.log.error('select data error, %s: %s' % (self.db_info, e))
            return False, None
        return True, data

    def TB_delete(self, delete_sql):
        "select data"
        logger.log.info('delete data: %s' % delete_sql)
        try:
            self.db.execute(delete_sql)
            self.conn.commit()
        except Exception as e:
            logger.log.error('delete data error, %s: %s' % (self.db_info, e))
            return False
        return True

    def TB_update(self, update_sql):
        "select data"
        logger.log.info('update table')
        try:
            self.db.execute(update_sql)
            self.conn.commit()
        except Exception as e:
            logger.log.error('update data error, %s: %s' % (self.db_info, e))
            return False
        return True

    def TB_truncate(self, table_name):
        "TRUNCATE table"
        # detect whether the table exist already
        if not self.TB_exist(table_name):
            logger.log.warn('truncate table, %s not exist' % table_name)
            return False

        truncate_sql = "TRUNCATE TABLE %s;" % table_name
        logger.log.info('truncate table %s' % table_name)

        try:
            self.db.execute(truncate_sql)
            self.conn.commit()
        except Exception as e:
            logger.log.error('truncate table error, %s: %s' % (self.db_info, e))
            return False
        return True
       
    def close(self):
        """ close the connection
        """
        try:
            self.db.close()
            self.conn.close()
            self.db = None
            self.conn = None
        except Exception as e:
            logger.log.error(e)
            return False

        return True


