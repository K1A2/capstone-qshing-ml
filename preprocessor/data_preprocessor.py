import sqlite3
import pandas as pd

import os
import argparse


class DataPreprocessor:
    def __init__(self,
                 args: argparse.Namespace):
        self.data_name = args.data_name
        self.data_path = os.path.join('./data', self.data_name)

        self.__data_init()

    def __data_init(self):
        con = sqlite3.connect(self.data_path)
        self.raw_data = pd.read_sql("SELECT * FROM data", con, index_col='id')

