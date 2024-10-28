import sqlite3
import pandas as pd

import os
import argparse

from logger.main_logger import MainLogger


class DataPreprocessor:
    def __init__(self,
                 args: argparse.Namespace):
        self.parallel = args.parallel
        if self.parallel == 1:
            self.gpu = args.gpu
        else:
            self.gpu = 0

        self.logger = MainLogger(args)
        self.data_name = args.data_name
        self.data_path = os.path.join('./data', self.data_name)
        self.logger.debug(f'data path: {self.data_path}', self.gpu)

        self.__data_init()

    def __data_init(self):
        con = sqlite3.connect(self.data_path)
        self.raw_data = pd.read_sql("SELECT url, html FROM data", con, index_col=None)

        self.logger.debug(f'raw data:\n{self.raw_data}', self.gpu)
