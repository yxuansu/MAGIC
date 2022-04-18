import sys
sys.path.append(r'../clip/')
import torch
import numpy as np
import progressbar
import os

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--image_file_prefix_path", type=str, help='the directory that stores images')
    parser.add_argument("--image_format", type=str, default='jpg')
    # save configuration
    parser.add_argument("--save_index_prefix", type=str, help='where to save the mips index')
    parser.add_argument("--save_index_name", type=str)
    parser.add_argument("--save_image_name_dict", type=str, 
        help="a json file that stores a dictory. The dictory contains mapping between mips index and image name")
    # inference configuration
    parser.add_argument("--batch_size", type=int, help="the batch size used to conduct inference with CLIP")
    return parser.parse_args()

def process_invalid_batch(model, batch_image, batch_name):
    bsz = len(batch_image)
    one_batch_vec_list, one_batch_image_name = [], []
    for idx in range(bsz):
        one_image, one_name = batch_image[idx], batch_name[idx]
        try:
            one_vec = model.compute_batch_index_image_features([one_image]).detach().cpu()
            one_vec = one_vec.numpy().reshape(-1)
            one_batch_vec_list.append(one_vec)
            one_batch_image_name.append(one_name)
        except:
            continue
    return one_batch_vec_list, one_batch_image_name

import copy
from PIL import Image
def open_batch_images(prefix_path, batch_name_list):
    batch_image_instance_list, batch_image_name_list = [], []
    for name in batch_name_list:
        one_image_path = prefix_path + '/' + name
        try:
            batch_image_instance_list.append(copy.deepcopy(Image.open(one_image_path)))
            batch_image_name_list.append(name)
        except:
            continue
    #assert len(batch_image_instance_list) == len(batch_name_list)
    assert len(batch_image_instance_list) == len(batch_image_name_list)
    return batch_image_instance_list, batch_image_name_list

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

    print ('Loading images...')
    #from utlis import parse_images
    #batched_image_instance_list, batched_image_name_list = \
    #parse_images(args.image_file_prefix_path, args.batch_size, image_format=args.image_format)
    from image_index_utlis import load_batch_image_names
    batched_image_name_list = load_batch_image_names(args.image_file_prefix_path, args.batch_size)
    print ('Images loaded!')

    image_vec_list, image_name_list = [], []
    batch_num = len(batched_image_name_list)
    print ('Number of image batches is {}'.format(batch_num))
    print ('Start image inference...')
    #batch_num = 10
    p = progressbar.ProgressBar(batch_num)
    p.start()
    with torch.no_grad():
        for p_idx in range(batch_num):
            p.update(p_idx)
            #one_image_batch = batched_image_instance_list[p_idx]
            one_image_batch, one_image_name_batch = \
            open_batch_images(args.image_file_prefix_path, batched_image_name_list[p_idx])
            #one_image_name_batch = batched_image_name_list[p_idx]
            try:
                one_batch_vec = model.compute_batch_index_image_features(one_image_batch).detach().cpu()
                one_batch_vec_list = one_batch_vec.unbind(dim=0)
                bsz = len(one_batch_vec_list)
                for k in range(bsz):
                    image_vec_list.append(one_batch_vec_list[k].numpy())
                    image_name_list.append(one_image_name_batch[k])
            except:
                one_batch_vec_list, one_batch_image_name = process_invalid_batch(model, one_image_batch, one_image_name_batch)
                if len(one_batch_vec_list) != 0:
                    bsz = len(one_batch_vec_list)
                    for k in range(bsz):
                        image_vec_list.append(one_batch_vec_list[k])
                        image_name_list.append(one_batch_image_name[k])
    p.finish()
    assert len(image_vec_list) == len(image_name_list)
    print ('Image inference completed!')

    assert len(image_name_list) == len(list(set(image_name_list))) # no duplication in the name list
    image_name_mapping_dict = {}
    for k in range(len(image_name_list)):
        image_name_mapping_dict[int(k)] = image_name_list[k]
    mapping_list_save_path = args.save_index_prefix + '/' + args.save_image_name_dict
    import json
    with open(mapping_list_save_path, 'w') as outfile:
        json.dump(image_name_mapping_dict, outfile, indent=4)
    print ('Mapping dictionary saved!')

    print ('Start buiding index...')
    index_save_path = args.save_index_prefix + '/' + args.save_index_name
    with open(index_save_path, 'w', encoding = 'utf8') as o:
        for vec in image_vec_list:
            one_text = ' '.join([str(num) for num in vec]).strip()
            o.writelines(one_text + '\n')
    print ('Index completed!')
