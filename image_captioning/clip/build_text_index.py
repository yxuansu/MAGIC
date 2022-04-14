import sys
import torch
import numpy as np
import progressbar
import os

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--text_file_path", type=str)
    # save configuration
    parser.add_argument("--save_index_prefix", type=str, help='where to save the mips index')
    parser.add_argument("--save_index_name", type=str)
    parser.add_argument("--save_mapping_dict_name", type=str, 
        help="a json file that stores a dictory. The dictory contains mapping between mips index and caption text")
    # inference configuration
    parser.add_argument("--batch_size", type=int, help="the batch size used to conduct inference with CLIP")
    return parser.parse_args()

def load_batch_text(text_file_path, batch_size):
    import json
    with open(text_file_path) as f:
        item_list = json.load(f)

    text_list = []
    for item in item_list:
        captions = item["captions"]
        for cap in captions:
            text_list.append(cap)
    print ('Number of text instances is {}'.format(len(text_list)))

    data_num = len(text_list)
    batch_num = data_num // batch_size
    batch_text_list = []
    s_idx, e_idx = 0, batch_size
    for p_idx in range(batch_num):
        one_batch_text_list = []
        for idx in range(s_idx, e_idx):
            one_batch_text_list.append(text_list[idx])
        batch_text_list.append(one_batch_text_list)
    return batch_text_list


import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    args = parse_config()
    device = torch.device('cuda')

    import os
    if os.path.exists(args.save_index_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(args.save_index_prefix, exist_ok=True)

    print ('Loading CLIP...')
    from clip import CLIP
    model = CLIP(args.clip_name)
    if cuda_available:
        model = model.cuda(device)
    model.eval()
    print ('CLIP loaded!')

    print ('Loading text data...')
    batch_text_list = load_batch_text(args.text_file_path, args.batch_size)
    print ('Text data loaded.')

    res_text_vec_list, res_text_list = [], []
    batch_num = len(batch_text_list)
    print ('Number of batches is {}'.format(batch_num))
    print ('Start inference...')
    p = progressbar.ProgressBar(batch_num)
    p.start()
    with torch.no_grad():
        for p_idx in range(batch_num):
            p.update(p_idx)
            one_text_batch = batch_text_list[p_idx]
            one_batch_vec = model.compute_batch_index_text_representation(one_text_batch).detach().cpu()
            one_batch_vec_list = one_batch_vec.unbind(dim=0)
            bsz = len(one_batch_vec_list)
            for k in range(bsz):
                res_text_vec_list.append(one_batch_vec_list[k].numpy())
                res_text_list.append(one_text_batch[k])
    p.finish()
    assert len(res_text_vec_list) == len(res_text_list)
    print ('Inference completed!')

    index_text_mapping_dict = {}
    for k in range(len(res_text_list)):
        index_text_mapping_dict[k] = res_text_list[k]
    mapping_list_save_path = args.save_index_prefix + '/' + args.save_mapping_dict_name
    import json
    with open(mapping_list_save_path, 'w') as outfile:
        json.dump(index_text_mapping_dict, outfile, indent=4)
    print ('Mapping dictionary saved!')

    print ('Start buiding index...')
    index_save_path = args.save_index_prefix + '/' + args.save_index_name
    with open(index_save_path, 'w', encoding = 'utf8') as o:
        for vec in res_text_vec_list:
            one_text = ' '.join([str(num) for num in vec]).strip()
            o.writelines(one_text + '\n')
    print ('Index completed!')
