import sqlite3
import pandas as pd

from html2text import HTML2Text
from bs4 import BeautifulSoup

import os
import argparse
from tqdm import tqdm

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

        urls, contents = self.__extract_data()

    def __extract_data(self):
        urls, contents = [], []

        converter = HTML2Text()

        for i in tqdm(range(self.raw_data.shape[0]), desc='extracting data'):
            html = self.raw_data.iloc[i]['html']
            contents.append(converter.handle(html))

            soup = BeautifulSoup(html, 'html.parser')
            url_parts = []
            for a_tag in soup.find_all('a', href=True):
                url_parts.append(a_tag['href'])
            for a_tag in soup.find_all('link', href=True):
                url_parts.append(a_tag['href'])
            urls.append(url_parts)

        self.logger.debug(f'urls: {len(urls)}\tcontents: {len(contents)}', self.gpu)
        return urls, contents
