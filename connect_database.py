# -*- coding = utf-8 -*-

import pandas as pd
import pymysql


class ConnectDatabase:
    """"""
    def __init__(self, db_info, sql):
        """"""
        self.db_info = db_info
        self.sql = sql

    def connect(self):
        host = self.db_info['host']
        port = self.db_info['port']
        username = self.db_info['username']
        password = self.db_info['password']
        database = self.db_info['database']

        try:
            conn = pymysql.connect(
                host=host,
                port=port,
                user=username,
                password=password,
                database=database
            )
            return conn
        except Exception as e:
            print(f'Error connecting to database:{e}')

    def get_data(self):
        if self.connect() is not None:
            with self.connect().cursor() as cursor:
                cursor.execute(self.sql)
                data = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                data = pd.DataFrame(list(data), columns=columns)
                if data.empty:
                    return data
            return data
        else:
            print('connection failed')
