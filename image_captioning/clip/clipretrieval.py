import json
import copy
import torch
import progressbar
import numpy as np
from PIL import Image

class CLIPIndex:
    def __init__(self, index_matrix_path, mapping_dict_path, clip):
        '''
            index_path: the pre-trained index
            mapping_dict_path: the pre-indexed mapping dictionary
            clip: the pre-trained clip model
        '''
        print ('Loading index...')
        self.index_matrix = self.normalization(self.load_matrix(index_matrix_path))
        print ('Index loaded.')
        print (self.index_matrix.shape)
        with open(mapping_dict_path) as f:
            self.mapping_dict = json.load(f)
        self.clip = clip

    def load_matrix(self, in_f):
        matrix_list = []
        with open(in_f, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_vec = [float(num) for num in l.strip('\n').split()]
                matrix_list.append(one_vec)
        return np.array(matrix_list)

    def normalization(self, matrix):
        '''
            matrix: num_instance x num_feature
        '''
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def get_image_representation(self, image_path):
        image_instance = Image.open(image_path)
        image_vec = self.clip.compute_batch_index_image_features([image_instance]).detach().cpu().numpy()
        image_vec = self.normalization(image_vec)
        return image_vec

    def search_text(self, image_path):
        image_vec = self.get_image_representation(image_path)
        sort_idx_list = np.matmul(image_vec, self.index_matrix.transpose())[0].argsort()[::-1]
        top_idx = sort_idx_list[0]
        return self.mapping_dict[str(top_idx)]


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_name", type=str)
    parser.add_argument("--test_image_prefix_path", type=str, help="the folder that stores all test images")
    parser.add_argument("--test_path", type=str)
    # index configuration
    parser.add_argument("--index_matrix_path", type=str)
    parser.add_argument("--mapping_dict_path", type=str)
    # save configuration
    parser.add_argument("--save_path_prefix", type=str, help="save the result in which directory")
    parser.add_argument("--save_name", type=str, help="the name of the saved file")
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
    save_name = args.save_name
    full_save_path = save_path_prefix + '/' + save_name
    print ('full save path is {}'.format(full_save_path))

    print ('Loading CLIP...')
    from clip import CLIP
    clip = CLIP(args.clip_name)
    if cuda_available:
        clip = clip.cuda(device)
    clip.eval()
    print ('CLIP loaded!')

    clipindex = CLIPIndex(args.index_matrix_path, args.mapping_dict_path, clip)

    print ('Loading data...')
    import json
    with open(args.test_path) as f:
        item_list = json.load(f)
    print ('Data loaded.')
    print ('Number of test instances is {}'.format(len(item_list)))

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

            image_full_path = args.test_image_prefix_path + '/' + one_test_dict['image_name']
            try:
                output_text = clipindex.search_text(image_full_path)
                one_res_dict['prediction'] = output_text
                result_list.append(one_res_dict)
            except:
                invalid_num += 1
                print ('invalid number is {}'.format(invalid_num))
                continue
        p.finish()
    print ('Inference completed!')

    import json
    with open(full_save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)

