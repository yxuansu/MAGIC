import os
import sys
import operator
from tqdm import tqdm
from operator import itemgetter
import torch
from torch import nn
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss_func import contrastive_loss

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')
class SimCTG(nn.Module):
    def __init__(self, model_name, sos_token, pad_token):
        super(SimCTG, self).__init__()
        from transformers import AutoTokenizer, GPT2LMHeadModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sos_token, self.sos_token_id = self.add_special_token(sos_token)
        print ('sos token is {}, sos token id is {}'.format(self.sos_token, self.sos_token_id))
        self.pad_token, self.pad_token_id = self.add_special_token(pad_token)
        print ('pad token is {}, pad token id is {}'.format(self.pad_token, self.pad_token_id))
        self.eos_token, self.eos_token_id = self.tokenizer.bos_token, self.tokenizer.bos_token_id
        print ('eos token is {}, eos token id is {}'.format(self.eos_token, self.eos_token_id))
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        print ('Resizing model embedding...')
        self.model.resize_token_embeddings(len(self.tokenizer)) 
        print ('Model embedding resized!')
        self.embed_dim = self.model.config.hidden_size

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

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels, margin):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = contrastive_loss(margin, cosine_scores, input_ids, self.pad_token_id, prefix_len=0)
        return mle_loss, cl_loss

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss = val_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        assert mle_loss.size() == torch.Size([bsz * seqlen])
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        # sum 
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    def parse_sentences(self, text, num_of_sentences_to_keep):
        item_list = text.split('.')
        res_list = item_list[:num_of_sentences_to_keep]
        if len(item_list) > num_of_sentences_to_keep:
            res_text = '.'.join(res_list).strip('.') + '.'
        else:
            res_text = '.'.join(res_list).strip('.').strip()
        return res_text

    def parse_generated_result(self, output, num_of_sentences_to_keep):
        output_text = self.tokenizer.decode(output)
        item_list = output_text.split(self.eos_token)
        full_text = self.eos_token.join(item_list[:2]).strip()
        full_text = self.parse_sentences(full_text, num_of_sentences_to_keep)
        generated_text = item_list[1].strip()
        generated_text = self.parse_sentences(generated_text, num_of_sentences_to_keep)
        return full_text, generated_text

    # decoding functions
    # ------------------------------------------------------- #

    def parse_output_token_list(self, output):
        output = output.tolist()
        res_list = []
        for token_id in output:
            if token_id == self.sos_token_id:
                continue
            elif token_id == self.eos_token_id:
                break
            else:
                res_list.append(token_id)
        text = self.tokenizer.decode(res_list).strip()
        return ' '.join(text.split()).strip()

    @torch.no_grad()
    def magic_search(self, input_ids, beam_width, alpha, decoding_len, beta, image_instance, clip, 
        clip_text_max_len):#, add_token_level_score=False):
        prefix_len = input_ids.size()[1]
        from utlis import PlugAndPlayContrastiveDecodingOneStepFast
        past_key_values, last_hidden_states, logits = None, None, None
        generated = [item for item in input_ids.tolist()]
        input_ids_for_class = input_ids.clone()

        image_embeds = clip.compute_image_representation_from_image_instance(image_instance)

        start_time = datetime.datetime.now()

        # the maximum supported length of generation for SimCTG is 256
        # to support longer generated length, you can re-train the SimCTG model with longer sequences
        decoding_len = decoding_len - prefix_len
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits, input_ids_for_class = \
            PlugAndPlayContrastiveDecodingOneStepFast(
                self.model, 
                input_ids, 
                prefix_len,
                beam_width, 
                alpha, 
                beta, 
                self.tokenizer,
                image_embeds, 
                clip, 
                clip_text_max_len,
                past_key_values,
                last_hidden_states,
                logits,
                first_step=step==0,
                input_ids_for_class=input_ids_for_class,
            )
        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000
        return self.parse_output_token_list(input_ids_for_class[0])

    def fast_contrastive_search(self, input_ids, beam_width, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        self.model.eval()
        from utlis import ContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0
        
        # fast mode
        prefix_len = input_ids.size()[1]
        batch_size, seqlen = input_ids.size()
        #generated = [[] for _ in range(batch_size)]
        generated = [item for item in input_ids.tolist()]
        past_key_values = None
        last_hidden_states = None
        logits = None
        decoding_len = decoding_len - prefix_len
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                first_step=step == 0,
            )
            tokens = input_ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
        return self.parse_output_token_list(torch.LongTensor(generated[0]))

    def top_k_sampling(self, input_ids, k, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=decoding_len, 
                            top_p=1.0,
                            top_k=k)
        return self.parse_output_token_list(output[0])

    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=decoding_len, 
                            top_p=nucleus_p,
                            top_k=0)
        return self.parse_output_token_list(output[0])
