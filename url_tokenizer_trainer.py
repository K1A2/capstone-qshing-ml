import pandas as pd

from tokenizers import CharBPETokenizer
from transformers import BertConfig, BertModel, BertTokenizerFast
import torch

import os
from tqdm import tqdm


df_data = pd.read_csv('./data/paper/concat.csv')
train_urls = df_data['url'].tolist()

train_file_path = './data/tokenizer/urls.txt'
with open(train_file_path, "w", encoding="utf-8") as f:
    for url in tqdm(train_urls, desc='save urls:'):
        f.write(url + "\n")

tokenizer = CharBPETokenizer()
tokenizer.train(files=train_file_path, vocab_size=90, min_frequency=1, special_tokens=[
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
])

tokenizer_path = './data/tokenizer'
tokenizer.save_model(tokenizer_path)
