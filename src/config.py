# -*- coding: utf-8 -*-

import transformers

BERT_PATH = 'bert-base-cased'
EPOCHS = 3
MAX_LEN = 512
BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH)