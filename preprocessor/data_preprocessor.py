import sqlite3
import pandas as pd

from transformers import BertTokenizer

from html2text import HTML2Text
from bs4 import BeautifulSoup
from langdetect import detect

import os
import argparse
from tqdm import tqdm
import time

from concurrent.futures import ThreadPoolExecutor

from logger.main_logger import MainLogger
import logger.utils as log_utils


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

        self.max_length = args.max_length
        self.data_workers = args.data_workers

        self.logger.debug(f'data path: {self.data_path}', self.gpu)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

        self.__data_init()

    def __data_init(self):
        con = sqlite3.connect(self.data_path)
        self.raw_data = pd.read_sql("SELECT url, html, label FROM data", con, index_col=None)

        self.logger.debug(f'raw data:\n{self.raw_data}', self.gpu)

        self.labels = self.raw_data.iloc[:]['label']
        urls, contents = self.__extract_data()

        self.urls = self.__url_tokenizer(urls)
        self.contents = self.__content_tokenizer(contents)

    def __extract_data_thread(self, thread_idx, start_idx, end_idx):
        self.logger.debug(f'extractor thread {thread_idx}: start', self.gpu)

        urls, contents = [], []
        converter = HTML2Text()

        for i in range(start_idx, end_idx):
            html = self.raw_data.iloc[i]['html']
            content = converter.handle(html)

            try:
                if detect(content) == 'en':
                    contents.append(content)
                else:
                    contents.append('')
            except:
                contents.append('')

            soup = BeautifulSoup(html, 'html.parser')
            url_parts = []
            for a_tag in soup.find_all('a', href=True):
                url_parts.append(a_tag['href'])
            for a_tag in soup.find_all('link', href=True):
                url_parts.append(a_tag['href'])
            urls.append(url_parts)

            if i % 2000 == 0:
                self.logger.debug(f'extractor thread {thread_idx}: {i + 1} finish', self.gpu)

        self.logger.debug(f'extractor thread {thread_idx}: finish', self.gpu)
        return urls, contents

    def __extract_data(self):
        self.logger.debug(f'extract data start', self.gpu)

        data_size = self.raw_data.shape[0]
        chunk_size = data_size // self.data_workers
        futures = []

        start = time.time()
        with ThreadPoolExecutor(max_workers=self.data_workers) as executor:
            for i in range(self.data_workers):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != self.data_workers - 1 else data_size
                futures.append(executor.submit(self.__extract_data_thread, i + 1, start_idx, end_idx))
        end = time.time()
        self.logger.debug(f'extract thread finished: {log_utils.time_to_str(end - start)}', self.gpu)

        urls, contents = [], []
        for future in futures:
            thread_urls, thread_contents = future.result()
            urls.extend(thread_urls)
            contents.extend(thread_contents)

        self.logger.debug(f'finish\turls: {len(urls)}\tcontents: {len(contents)}', self.gpu)
        return urls, contents

    def __url_tokenizer(self, urls):
        tokens = []
        for url in tqdm(urls, desc='url tokenization'):
            token = []
            for u in url:
                token.append('[CLS]')
                token.extend(list(u))
            token = ' '.join(token)
            tokens.append(token)
        tokenized_output = self.tokenizer(tokens, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)
        return tokenized_output

    def __content_tokenizer(self, contents):
        tokens = ['[CLS]']
        for text in tqdm(contents, desc='content tokenization'):
            tokens.extend(self.tokenizer.tokenize(text))
            tokens.append('[SEP]')

        tokenized_output = self.tokenizer(tokens, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)
        return tokenized_output

    def get_data(self):
        return self.urls, self.contents, self.labels
