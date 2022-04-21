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

import logging
logging.getLogger('transformers.generation_utils').disabled = True

def perform_inference(args, data, full_save_path, generation_model, classifier, cuda_available, device):
    print ('----------------------------------------------------------------')
    print ('Start inference...')
    num_class, number_of_instance_to_generate_per_class, decoding_len = args.num_class, \
    args.number_of_instance_to_generate_per_class, args.decoding_len
    k, alpha, beta = args.k, args.alpha, args.beta
    eos_token = '<|endoftext|>'
    test_num = len(data.prefix_token_id_list)
    #test_num = 10
    result_list = []
    p = progressbar.ProgressBar(test_num)
    p.start()
    with torch.no_grad():
        for index in range(test_num):
            p.update(index)
            one_res_dict = inference_one_instance(data, index, num_class, number_of_instance_to_generate_per_class, 
                decoding_len, k, alpha, beta, eos_token, generation_model, classifier, cuda_available, device)
            result_list.append(one_res_dict)
        p.finish()
    import json
    with open(full_save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)
    print ('Inference completed.')

def inference_one_instance(args, data, index, generation_model, cuda_available, device):
    '''
        data: the inference dataclass;
        index: the index of specific instance;
        generation_model: the underlying generative model;
        cuda_available: whether GPU is available; 
        device: if GPU is available, then which device to use;
    '''
    eos_token = generation_model.tokenizer.bos_token # the end of sequence token of the generation model

    res_dict = {}
    res_dict['prefix_text'] = data.prefix_text_list[index]
    res_dict['reference_continuation_text'] = data.reference_continuation_text_list[index]

    generated_dict = {}
    input_ids = data.prefix_token_id_list[index]
    input_ids = torch.LongTensor(input_ids).view(1,-1)
    if cuda_available:
        input_ids = input_ids.cuda(device)

    decoding_method, decoding_len = args.decoding_method, args.decoding_len
    beam_width, top_k, nucleus_p, k, alpha = args.beam_width, args.top_k, args.nucleus_p, args.k, args.alpha
    number_of_instance_to_generate_per_method = args.number_of_instance_to_generate_per_method

    for instance_idx in range(number_of_instance_to_generate_per_method):
        generated_dict[instance_idx] = {}
        if decoding_method == 'greedy':
            output, _ = generation_model.greedy_search(input_ids, decoding_len, eos_token)
            _, one_generated_continuation_text = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
            one_generated_full_text = data.prefix_text_list[index] + ' ' + one_generated_continuation_text

        elif args.decoding_method == 'beam':
            output, _ = generation_model.beam_search(input_ids, beam_width, decoding_len, eos_token)
            _, one_generated_continuation_text = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
            one_generated_full_text = data.prefix_text_list[index] + ' ' + one_generated_continuation_text

        elif args.decoding_method == 'top-k':
            output, _ = generation_model.top_k_sampling(input_ids, top_k, decoding_len, eos_token)
            _, one_generated_continuation_text = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
            one_generated_full_text = data.prefix_text_list[index] + ' ' + one_generated_continuation_text

        elif args.decoding_method == 'nucleus':
            output, _ = generation_model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token)
            _, one_generated_continuation_text = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
            one_generated_full_text = data.prefix_text_list[index] + ' ' + one_generated_continuation_text

        elif args.decoding_method == 'typical':
            output, _ = generation_model.typical_sampling(input_ids, args.typical_mass, decoding_len, eos_token)
            _, one_generated_continuation_text = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
            one_generated_full_text = data.prefix_text_list[index] + ' ' + one_generated_continuation_text

        elif args.decoding_method == 'contrastive':
            output, _ = generation_model.fast_contrastive_search(input_ids, k, alpha, decoding_len, eos_token)
            _, one_generated_continuation_text = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
            one_generated_full_text = data.prefix_text_list[index] + ' ' + one_generated_continuation_text

        else:
            raise Exception('Wrong decoding method!!!')

        generated_dict[instance_idx]['generated_continuation_text'] = one_generated_continuation_text
        generated_dict[instance_idx]['generated_full_text'] = one_generated_full_text
    res_dict['generated_result'] = generated_dict
    return res_dict

def parse_config():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--language_model_code_path", type=str, help="where is the code of language model located")
    # model and data configuration
    parser.add_argument("--language_model_name", type=str, help="name of pre-trained language model")
    parser.add_argument("--test_path", type=str)
    # decoding configuration
    parser.add_argument("--num_of_inference_instances", type=int, help="how many inference instances to consider")
    parser.add_argument("--decoding_len", type=int, help='maximum length of (prefix + generated continuation)')
    parser.add_argument("--number_of_instance_to_generate_per_method", type=int, 
        help="number of instances to generate per decoding method")
    # decoding configuration
    parser.add_argument("--decoding_method", type=str, help='e.g. greedy, beam, top-k, nucleus, contrastive')
    # beam search configuration
    parser.add_argument("--beam_width", type=int, default=-1, help="beam width for beam search")
    # top-k sampling configuration
    parser.add_argument("--top_k", type=int, default=-1, help="k for top-k sampling")
    # nucleus sampling configuration
    parser.add_argument("--nucleus_p", type=float, default=-1.0, help='p for nucleus sampling')
    # typical sampling configuration
    parser.add_argument("--typical_mass", type=float, default=-1.0, help='mass for typical sampling')
    # magic configuration
    parser.add_argument("--k", type=int, default=-1, help='k for contrastive search')
    parser.add_argument("--alpha", type=float, default=-1.0, help="alpha for contrastive search")
    # save configuration
    parser.add_argument("--save_path_prefix", type=str, help="save the result in which directory")
    return parser.parse_args()

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
    if args.decoding_method == 'greedy':
        save_name = 'greedy_result.json'
    elif args.decoding_method == 'beam':
        save_name = 'beam_{}_result.json'.format(args.beam_width)
    elif args.decoding_method == 'top-k':
        save_name = 'top_{}_result.json'.format(args.top_k)
    elif args.decoding_method == 'nucleus':
        save_name = 'nucleus_{}_result.json'.format(args.nucleus_p)
    elif args.decoding_method == 'typical':
        save_name = 'typical_{}_result.json'.format(args.typical_mass)
    elif args.decoding_method == 'contrastive':
        save_name = 'contrastive_k_{}_alpha_{}_result.json'.format(args.k, args.alpha)
    else:
        raise Exception('Wrong decoding method!!!')
    full_save_path = save_path_prefix + '/' + save_name
    print ('full save path is {}'.format(full_save_path))

    print ('Loading data...')
    from inference_dataclass import Data
    data = Data(args.language_model_name, args.test_path)
    print ('Data loaded.')

    print ('Loading off-the-shelf language model...')
    import sys
    from transformers import AutoTokenizer
    sys.path.append(args.language_model_code_path)
    from simctg import SimCTG
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name)
    generation_model = SimCTG(args.language_model_name, tokenizer.pad_token_id)
    if cuda_available:
        generation_model = generation_model.to(device)
    generation_model.eval()
    print ('Language model loaded.')

    result_dict = {}
    result_dict['decoding_method'] = args.decoding_method
    result_dict['evaluation_result'] = []
    print ('----------------------------------------------------------------')
    print ('Start inference with {} method...'.format(args.decoding_method))
    with torch.no_grad():
        #test_num = len(data.prefix_token_id_list)
        #test_num = 10
        test_num = args.num_of_inference_instances
        print ('Number of inference instances is {}'.format(test_num))
        p = progressbar.ProgressBar(test_num)
        p.start()
        for p_idx in range(test_num):
            p.update(p_idx)
            one_res_dict = inference_one_instance(args, data, p_idx, generation_model, cuda_available, device)
            result_dict['evaluation_result'].append(one_res_dict)
        p.finish()
    print ('Inference completed!')

    import json
    with open(full_save_path, 'w') as outfile:
        json.dump(result_dict, outfile, indent=4)
