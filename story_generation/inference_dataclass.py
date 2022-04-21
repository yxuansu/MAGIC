import random
import torch
import numpy as np
import progressbar
from torch.nn.utils import rnn

class Data:
    def __init__(self, model_name, test_path):
        '''
            test_path: data path to validate the result
        '''
        from transformers import GPT2TokenizerFast
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

        self.prefix_token_id_list, self.prefix_text_list, self.reference_continuation_text_list = \
        self.process_one_file(test_path)
        print ('Evaluation number is {}'.format(len(self.prefix_token_id_list)))

    def process_one_file(self, path):
        print ('Processing {}'.format(path))
        prefix_token_id_list, prefix_text_list, reference_continuation_text_list = [], [], []

        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
        n = len(lines)
        print (n)
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            text = lines[i].strip('\n')
            self.process_one_text(text, prefix_token_id_list, prefix_text_list, reference_continuation_text_list)
        p.finish()
        print ('{} processed!'.format(path))
        return prefix_token_id_list, prefix_text_list, reference_continuation_text_list

    def process_one_text(self, text, prefix_token_id_list, prefix_text_list, reference_continuation_text_list):
        item_list = text.strip('\n').split('\t')
        prefix_text = item_list[0] + ' ' + self.tokenizer.eos_token
        prefix_tokens = self.tokenizer.tokenize(prefix_text)
        prefix_id_list = self.tokenizer.convert_tokens_to_ids(prefix_tokens)
        prefix_len = len(prefix_id_list)

        reference_text = item_list[1].strip()
        reference_tokens = self.tokenizer.tokenize(reference_text)

        reference_token_id_list = self.tokenizer.convert_tokens_to_ids(reference_tokens)
        reference_continuation_text = self.tokenizer.decode(reference_token_id_list).strip()
        prefix_token_id_list.append(prefix_id_list)
        prefix_text_list.append(prefix_text)
        reference_continuation_text_list.append(reference_continuation_text)