import torch

import sqlite3
import pandas as pd

from transformers import BertTokenizer

from html2text import HTML2Text
from bs4 import BeautifulSoup
from langdetect import detect
import re

import os
import argparse
from tqdm import tqdm
import time

from concurrent.futures import ThreadPoolExecutor

from logger.main_logger import MainLogger
import logger.utils as log_utils

from sklearn.model_selection import train_test_split
import pickle


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, urls, contents, labels):
        self.urls = urls
        self.contents = contents
        self.labels = labels

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        return (
            {
                'url_input_ids': self.urls['input_ids'][idx].squeeze(0),
                'url_attention_mask': self.urls['attention_mask'][idx].squeeze(0),
                'html_input_ids': self.contents['input_ids'][idx].squeeze(0),
                'html_attention_mask': self.contents['attention_mask'][idx].squeeze(0)
            },
            self.labels.iloc[idx].squeeze(0)
        )


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
        self.batch_size = args.batch_size
        self.num_worker = args.loader_worker
        self.split_ratio = args.split_ratio

        self.logger.debug(f'data path: {self.data_path}', self.gpu)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

        self.save_urls_path = os.path.join('./data', 'urls.plck')
        self.save_contents_path = os.path.join('./data', 'contents.plck')
        self.save_labels_path = os.path.join('./data', 'labels.plck')
        self.save_train_val_idx = os.path.join('./data', 'tvidx.plck')
        
        if args.save_data == 1:
            try:
                with open(self.save_urls_path, 'rb') as f:
                    urls = pickle.load(f)
                with open(self.save_contents_path, 'rb') as f:
                    contents = pickle.load(f)
                with open(self.save_labels_path, 'rb') as f:
                    labels = pickle.load(f)
                with open(self.save_train_val_idx, 'rb') as f:
                    train_val_idx = pickle.load(f)
            except:
                self.logger.debug(f'preprocessed data file not found', self.gpu)
                self.__data_init()
            self.urls = urls
            self.contents = contents
            self.labels = labels
            self.train_idx = train_val_idx['train_idx']
            self.val_idx = train_val_idx['val_idx']
        else:
            self.__data_init()

    def __data_init(self):
        con = sqlite3.connect(self.data_path)
        self.raw_data = pd.read_sql("SELECT url, html, label FROM data", con, index_col=None)
        
        label_counts = self.raw_data['label'].value_counts()
        count_f = label_counts.get(0, 0)
        count_p = label_counts.get(1, 0)

        self.logger.debug(f'raw data: {self.raw_data}\t0: {count_f}\t1: {count_p}', self.gpu)
        
        if count_f > count_p:
            df_label_0 = self.raw_data[self.raw_data['label'] == 0].sample(n=count_p, random_state=42)
            df_label_1 = self.raw_data[self.raw_data['label'] == 1]
        elif count_p > count_f:
            df_label_1 = self.raw_data[self.raw_data['label'] == 1].sample(n=count_f, random_state=42)
            df_label_0 = self.raw_data[self.raw_data['label'] == 0]
        else:
            df_label_0 = self.raw_data[self.raw_data['label'] == 0]
            df_label_1 = self.raw_data[self.raw_data['label'] == 1]

        self.raw_data = pd.concat([df_label_0, df_label_1]).sample(frac=1, random_state=42)
        
        self.logger.debug(f'balanced raw data: {self.raw_data.shape[0]}', self.gpu)

        self.labels = self.raw_data.iloc[:]['label']
        urls, contents = self.__extract_data()

        self.urls = self.__url_tokenizer(urls)
        self.contents = self.__content_tokenizer(contents)
        
        train_idx, val_idx = [], []
        for cls in range(2):
            cls_idx = [i for i, t in enumerate(self.labels) if t == cls]
            cls_train_idx, cls_val_idx = train_test_split(cls_idx, test_size=self.split_ratio, random_state=42)
            train_idx.extend(cls_train_idx)
            val_idx.extend(cls_val_idx)
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        
        with open(self.save_urls_path, 'wb') as f:
            pickle.dump(self.urls, f, pickle.HIGHEST_PROTOCOL)
        with open(self.save_contents_path, 'wb') as f:
            pickle.dump(self.contents, f, pickle.HIGHEST_PROTOCOL)
        with open(self.save_labels_path, 'wb') as f:
            pickle.dump(self.labels, f, pickle.HIGHEST_PROTOCOL)
        with open(self.save_train_val_idx, 'wb') as f:
            pickle.dump({'train_idx': self.train_idx, 'val_idx': self.val_idx}, f, pickle.HIGHEST_PROTOCOL)

    def __extract_data_thread(self, thread_idx, start_idx, end_idx):
        self.logger.debug(f'extractor thread {thread_idx}: start', self.gpu)

        urls, contents = [], []
        converter = HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True 
        converter.ignore_tables = True

        for i in range(start_idx, end_idx):
            html = self.raw_data.iloc[i]['html']
            content = converter.handle(html)

            sentences = re.split(r'(?<=[.!?]) +', content)
            
            contents_parts = []
            for s in sentences:
                try:
                    if detect(s) == 'en':
                        contents_parts.append(s)
                    # else:
                    #     contents.append('')
                except:
                    pass
                    # contents.append('')
            contents.append(contents_parts)

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
        for sentences in tqdm(contents, desc='content tokenization'):
            token = []
            for idx, s in enumerate(sentences):
                if idx != 0:
                    token.append('[SEP]')
                token.extend(s)
            token = ''.join(token)
            tokens.append(token)

        tokenized_output = self.tokenizer(tokens, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)
        return tokenized_output

    def get_data(self):
        dataset = MultimodalDataset(self.urls, self.contents, self.labels)
        
        train_dataset = torch.utils.data.Subset(dataset, self.train_idx)
        val_dataset = torch.utils.data.Subset(dataset, self.val_idx)
        
        self.logger.debug(f'tain: {len(self.train_idx)} valid: {len(self.val_idx)}')
        # self.logger.debug(f'tain: {self.train_idx} valid: {self.val_idx}')
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_worker, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=True, pin_memory=True, num_workers=0)
        return train_loader, valid_loader
