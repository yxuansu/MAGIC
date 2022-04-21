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

def inference_one_instance(args, data, index, imageindex, clip, generation_model, cuda_available, device):
    '''
        data: the inference dataclass;
        index: the index of specific instance;
        imageindex: the image index
        generation_model: the underlying generative model;
        cuda_available: whether GPU is available; 
        device: if GPU is available, then which device to use;
    '''
    eos_token = generation_model.tokenizer.bos_token # the end of sequence token of the generation model

    res_dict = {}
    res_dict['prefix_text'] = data.prefix_text_list[index]
    res_dict['reference_continuation_text'] = data.reference_continuation_text_list[index]

    '''
        the original clip text encoder only supports text shorter than 77 tokens. To make sure the length of
        the text does not surpass the maximum length, we truncate it with the last 60 tokens during decoding
    '''
    clip_text_max_len = 60 

    # retrieve image from the index
    image_name_list, image_instance_list = \
    imageindex.search_image(data.prefix_text_list[index], top_k=args.number_of_instance_to_generate_per_method)

    generated_dict = {}
    input_ids = data.prefix_token_id_list[index]
    input_ids = torch.LongTensor(input_ids).view(1,-1)
    if cuda_available:
        input_ids = input_ids.cuda(device)

    decoding_len = args.decoding_len
    k, alpha, beta = args.k, args.alpha, args.beta
    number_of_instance_to_generate_per_method = args.number_of_instance_to_generate_per_method
    assert len(image_name_list) == number_of_instance_to_generate_per_method
    assert len(image_instance_list) == number_of_instance_to_generate_per_method

    for instance_idx in range(number_of_instance_to_generate_per_method):
        generated_dict[instance_idx] = {}
        one_image_name, one_image_instance = image_name_list[instance_idx], \
        image_instance_list[instance_idx]
        generated_dict[instance_idx]['image_name'] = one_image_name

        output, _ = generation_model.magic_search(input_ids, k, alpha, decoding_len, beta, one_image_instance, 
            clip, clip_text_max_len, eos_token)
        _, one_generated_continuation_text = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
        one_generated_full_text = data.prefix_text_list[index] + ' ' + one_generated_continuation_text

        generated_dict[instance_idx]['generated_continuation_text'] = one_generated_continuation_text
        generated_dict[instance_idx]['generated_full_text'] = one_generated_full_text
    res_dict['generated_result'] = generated_dict
    return res_dict

def parse_config():
    parser = argparse.ArgumentParser()
    # language model path setting
    parser.add_argument("--language_model_code_path", type=str, help="where is the code of language model located")
    # model and data configuration
    parser.add_argument("--language_model_name", type=str, help="name of pre-trained language model")
    # index path setting
    parser.add_argument("--image_index_code_path", type=str, help="where is the code of image index located")
    parser.add_argument("--clip_path", type=str, help="where is the clip code located")
    parser.add_argument("--clip_name", type=str)
    parser.add_argument("--image_index_matrix_path", type=str, help="the file stores the representations of image database")
    parser.add_argument("--image_mapping_dict_path", type=str, 
        help="the dictionary that stores mapping between image index and image name")
    parser.add_argument("--image_folder_prefix_path", type=str, help="the folder that stores all images")
    # test data path
    parser.add_argument("--test_path", type=str)
    # decoding configuration
    parser.add_argument("--num_of_inference_instances", type=int, help="how many inference instances to consider")
    parser.add_argument("--decoding_len", type=int, help='maximum length of (prefix + generated continuation)')
    parser.add_argument("--number_of_instance_to_generate_per_method", type=int, 
        help="number of instances to generate per decoding method")
    # magic configuration
    parser.add_argument("--k", type=int, default=-1, help='k for magic search')
    parser.add_argument("--alpha", type=float, default=-1.0, help="alpha for magic search")
    parser.add_argument("--beta", type=float, default=-1.0, help="beta for magic search")
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
    save_name = 'magic_k_{}_alpha_{}_beta_{}_result.json'.format(args.k, args.alpha, args.beta)
    full_save_path = save_path_prefix + '/' + save_name
    print ('full save path is {}'.format(full_save_path))

    print ('Loading data...')
    from inference_dataclass import Data
    data = Data(args.language_model_name, args.test_path)
    print ('Data loaded.')

    import sys
    sys.path.append(args.clip_path)
    print ('Loading CLIP...')
    from clip import CLIP
    clip = CLIP(args.clip_name)
    if cuda_available:
        clip = clip.cuda(device)
    clip.eval()
    print ('CLIP loaded!')
    print ('Loading Image Index...')
    import sys
    sys.path.append(args.image_index_code_path)
    from imageindex import ImageIndex
    index = ImageIndex(args.image_index_matrix_path, args.image_mapping_dict_path, 
        args.image_folder_prefix_path, clip)
    print ('Image Index Loaded!')

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
    result_dict['evaluation_result'] = []
    print ('----------------------------------------------------------------')
    with torch.no_grad():
        test_num = args.num_of_inference_instances
        print ('Number of inference instances is {}'.format(test_num))
        p = progressbar.ProgressBar(test_num)
        p.start()
        for p_idx in range(test_num):
            p.update(p_idx)
            one_res_dict = inference_one_instance(args, data, p_idx, index, clip, generation_model, 
                cuda_available, device)
            result_dict['evaluation_result'].append(one_res_dict)
        p.finish()
    print ('Inference completed!')

    import json
    with open(full_save_path, 'w') as outfile:
        json.dump(result_dict, outfile, indent=4)
