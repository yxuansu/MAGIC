import json
import copy
import numpy as np
from PIL import Image

class ImageIndex:
    def __init__(self, index_path, mapping_dict_path, image_folder_prefix_path, clip):
        '''
            index_path: the pre-trained index
            mapping_dict_path: the pre-indexed mapping dictionary
            image_folder_prefix_path: the prefix path where to find images
            clip: the pre-trained clip model
        '''
        print ('Loading index...')
        self.index_matrix = self.normalization(self.load_matrix(index_path))
        print ('Index loaded.')
        print (self.index_matrix.shape)
        with open(mapping_dict_path) as f:
            self.mapping_dict = json.load(f)
        self.image_folder_prefix_path = image_folder_prefix_path
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

    def parse_prompt(self, text):
        '''
            process the prompt text;
            e.g. 
            input: 
                [ WP ] While searching the site of a disaster , you find your own corpse .
            output:
                While searching the site of a disaster , you find your own corpse .
        '''
        eos_token = '<|endoftext|>'
        text = text.strip(eos_token).strip()
        left_bracket_idx, right_bracket_idx = -1, -1
        for idx in range(len(text)):
            char = text[idx]
            if char == '[' and left_bracket_idx == -1: # first [ is met
                left_bracket_idx = idx
            elif char == ']' and right_bracket_idx == -1: # first ] is met
                right_bracket_idx = idx
            else:
                pass
        res_text = ''
        remove = False
        if left_bracket_idx > -1 and right_bracket_idx > left_bracket_idx:
            if right_bracket_idx - left_bracket_idx <= 6:
                remove = True
            else:
                pass

        for idx in range(len(text)):
            if remove:
                if idx >= left_bracket_idx and idx <= right_bracket_idx:
                    continue
                else:
                    res_text += text[idx]
            else:
                res_text += text[idx]
        res_text = res_text.strip()
        res_text = ' '.join(res_text.split()).strip()
        return res_text

    def get_text_representation(self, text):
        text = self.parse_prompt(text)
        text_vec = self.clip.compute_batch_index_text_representation([text]).detach().cpu().numpy()
        text_vec = self.normalization(text_vec)
        return text_vec

    def search_image(self, text, top_k):
        '''
            text: the story prompt
            top_k: number of images to retrieve
        '''
        text_vec = self.get_text_representation(text)
        sort_idx_list = np.matmul(text_vec, self.index_matrix.transpose())[0].argsort()[::-1]
        image_name_list, image_instance_list = [], []
        for idx in sort_idx_list[:top_k]:
            one_name = self.mapping_dict[str(idx)]
            image_name_list.append(one_name)
            one_full_path = self.image_folder_prefix_path + '/' + one_name
            image_instance_list.append(copy.deepcopy(Image.open(one_full_path)))
        return image_name_list, image_instance_list
    