# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar
from PIL import Image

import logging
logging.getLogger('transformers.generation_utils').disabled = True

def parse_config():
    parser = argparse.ArgumentParser()
    # language model path setting
    parser.add_argument("--language_model_code_path", type=str, help="where is the code of language model located")
    # model and data configuration
    parser.add_argument("--language_model_name", type=str, help="name of pre-trained language model")
    # test data path
    parser.add_argument("--test_path", type=str)
    # decoding configuration
    parser.add_argument("--decoding_method", type=str)
    # top-k sampling configuration
    parser.add_argument("--top_k", type=int, default = -1)
    # nucleus sampling configuration
    parser.add_argument("--nucleus_p", type=float, default=-1.0) 
    # contrastive search configuration
    parser.add_argument("--k", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=-1.0)
    # decoding length configuration
    parser.add_argument("--decoding_len", type=int, default=16, help='maximum length of (prefix + generated continuation)')
    # save configuration
    parser.add_argument("--save_path_prefix", type=str, help="save the result in which directory")
    parser.add_argument("--save_name", type=str, help="the name of the saved file")
    return parser.parse_args()

def get_prompt_id(text, tokenizer):
    text = text
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.LongTensor(input_ids).view(1,-1)
    return input_ids

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    args = parse_config()
    device = torch.device('cuda')

    save_path_prefix = args.save_path_prefix
    import os
    if os.path.exists(save_path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)
    # parse save name
    save_name = args.save_name
    full_save_path = save_path_prefix + '/' + save_name
    print ('full save path is {}'.format(full_save_path))

    print ('Loading data...')
    import json
    with open(args.test_path) as f:
        item_list = json.load(f)
    print ('Data loaded.')
    print ('Number of test instances is {}'.format(len(item_list)))

    print ('Loading off-the-shelf language model...')
    import sys
    sys.path.append(args.language_model_code_path)
    from simctg import SimCTG
    sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
    clip_text_max_len = 60
    generation_model = SimCTG(args.language_model_name, sos_token, pad_token)
    if cuda_available:
        generation_model = generation_model.to(device)
    generation_model.eval()
    print ('Language model loaded.')

    result_list = []
    invalid_num = 0
    print ('----------------------------------------------------------------')
    with torch.no_grad():
        test_num = len(item_list)
        #test_num = 10
        print ('Number of inference instances is {}'.format(test_num))
        p = progressbar.ProgressBar(test_num)
        p.start()
        for p_idx in range(test_num):
            p.update(p_idx)
            one_test_dict = item_list[p_idx]

            one_res_dict = {
                'split':one_test_dict['split'],
                'image_name':one_test_dict['image_name'],
                #'file_path':one_test_dict['file_path'],
                'captions':one_test_dict['captions']
            }

            input_ids = get_prompt_id(sos_token, generation_model.tokenizer)
            if cuda_available:
                input_ids = input_ids.cuda(device)

            if args.decoding_method == 'topk':
                assert args.top_k > 0
                output_text = generation_model.top_k_sampling(input_ids, args.top_k, args.decoding_len)
            elif args.decoding_method == 'nucleus':
                assert args.nucleus_p > 0
                output_text = generation_model.nucleus_sampling(input_ids, args.nucleus_p, args.decoding_len)
            elif args.decoding_method == 'contrastive':
                assert args.k > 0
                assert args.alpha > 0
                output_text = generation_model.fast_contrastive_search(input_ids, args.k, args.alpha, args.decoding_len)
            else:
                raise Exception("Wrong decoding configuration!!!")
            one_res_dict['prediction'] = output_text
            result_list.append(one_res_dict)
        p.finish()
    print ('Inference completed!')

    import json
    with open(full_save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)
