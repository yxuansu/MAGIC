import json
import random
import torch
import numpy as np
import progressbar
from torch.nn.utils import rnn

class Data:
    def __init__(self, model_name, train_path, dev_path, test_path, max_len, 
        sos_token, pad_token, add_eos_token_to_data):
        '''
            model_name: gpt2
            train_path: training data path
            dev_path: validation data path
            test_path: test data path 
            max_len: maximum length for training sequences 
            sos_token: initialized sos token <-start_of_text->
            pad_token: used to pad the sequences <-pad->
            add_eos_token_to_data: whether we want to the model learn to generate eos token;
                if so, the model could automatically stop generation by generating eos token
        '''
        from transformers import GPT2TokenizerFast
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.sos_token, self.sos_token_id = self.add_special_token(sos_token)
        print ('sos token is {}, sos token id is {}'.format(self.sos_token, self.sos_token_id))
        self.pad_token, self.pad_token_id = self.add_special_token(pad_token)
        print ('pad token is {}, pad token id is {}'.format(self.pad_token, self.pad_token_id))
        self.eos_token, self.eos_token_id = self.tokenizer.bos_token, self.tokenizer.bos_token_id
        print ('eos token is {}, eos token id is {}'.format(self.eos_token, self.eos_token_id))
        self.add_eos_token_to_data = add_eos_token_to_data

        self.max_len = max_len
        self.train_token_list, self.train_token_id_list = self.process_one_file(train_path)
        self.dev_token_list, self.dev_token_id_list = self.process_one_file(dev_path)
        self.test_token_list, self.test_token_id_list = self.process_one_file(test_path)
        self.train_num, self.dev_num, self.test_num = len(self.train_token_list), len(self.dev_token_list), \
        len(self.test_token_list)
        print ('train number:{}, dev number:{}, test number:{}'.format(self.train_num, self.dev_num, self.test_num))

        self.train_idx_list = [i for i in range(self.train_num)]
        random.shuffle(self.train_idx_list)
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.test_idx_list = [j for j in range(self.test_num)]
        self.dev_current_idx, self.test_current_idx = 0, 0

    def add_special_token(self, special_token):
        if special_token in self.tokenizer.vocab:
            print (special_token + ' token exists.')
        else:
            print ('Add token to the tokenizer.')
            print ('Original vocabulary size is {}'.format(len(self.tokenizer)))
            self.tokenizer.add_tokens([special_token])
            print ('Vocabulary size after extension is {}'.format(len(self.tokenizer)))
            assert len(self.tokenizer.convert_tokens_to_ids([special_token])) == 1
        special_token_id = self.tokenizer.convert_tokens_to_ids([special_token])[0]
        return special_token, special_token_id

    def process_one_file(self, path):
        print ('Processing {}'.format(path))
        with open(path) as f:
            item_list = json.load(f)
        lines = []
        for item in item_list:
            captions_list = item['captions']
            for one_caption in captions_list:
                lines.append(one_caption.strip())

        res_token_list, res_token_id_list = [], []
        n = len(lines)
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            text = lines[i].strip('\n')
            self.process_one_text(text, res_token_list, res_token_id_list)
        p.finish()
        print ('{} processed!'.format(path))
        return res_token_list, res_token_id_list

    def process_one_text(self, text, res_token_list, res_token_id_list):
        tokens = self.tokenizer.tokenize(text, max_length=self.max_len, truncation=True)
        if len(tokens) <= 1: # filter out too short sequence
            return
        tokens = [self.sos_token] + tokens[:self.max_len]
        if self.add_eos_token_to_data:
            tokens = tokens + [self.eos_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        res_token_list.append(tokens)
        res_token_id_list.append(token_ids)
        return

    def pad_batch(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask

    def process_output(self, batch_tgt_id_list):
        batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list) # padded target sequence
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch(self, batch_id_list):
        batch_input, batch_labels = self.process_output(batch_id_list)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_input, batch_labels

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_id_list, batch_token_list = [], []

        for idx in batch_idx_list:
            batch_id_list.append(self.train_token_id_list[idx])
            batch_token_list.append(self.train_token_list[idx])
        batch_input_tensor, batch_labels = self.parse_batch(batch_id_list)
        return batch_input_tensor, batch_labels, batch_token_list

    def get_next_validation_batch(self, batch_size, mode):
        batch_id_list, batch_token_list = [], []
        if mode == 'dev':
            curr_select_idx, instance_num = self.dev_current_idx, self.dev_num
            tgt_token_id_list, tgt_token_list = self.dev_token_id_list, self.dev_token_list
        elif mode == 'test':
            curr_select_idx, instance_num = self.test_current_idx, self.test_num
            tgt_token_id_list, tgt_token_list = self.test_token_id_list, self.test_token_list
        else:
            raise Exception('Wrong Validation Mode!!!')

        if curr_select_idx + batch_size < instance_num:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                batch_id_list.append(tgt_token_id_list[curr_idx])
                batch_token_list.append(tgt_token_list[curr_idx])
            if mode == 'dev':
                self.dev_current_idx += batch_size
            else:
                self.test_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                if curr_idx > instance_num - 1: 
                    curr_idx = 0
                    if mode == 'dev':
                        self.dev_current_idx = 0
                    else:
                        self.test_current_idx = 0
                batch_id_list.append(tgt_token_id_list[curr_idx])
                batch_token_list.append(tgt_token_list[curr_idx])
            if mode == 'dev':
                self.dev_current_idx = 0
            else:
                self.test_current_idx = 0
        batch_input_tensor, batch_labels = self.parse_batch(batch_id_list)
        return batch_input_tensor, batch_labels, batch_token_list
