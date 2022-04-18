from os import listdir
from PIL import Image
import progressbar
import copy

'''
def load_image_names(prefix_path, image_format='jpg'):
    filename_list = listdir(prefix_path)
    image_name_list = []
    for filename in filename_list:
        if filename.endswith(image_format):
            image_name_list.append(filename)
    image_name_list = list(set(image_name_list))
    return image_name_list
'''

def load_image_instances(prefix_path, image_name_list, batch_size):
    image_instance_list = []
    pp = progressbar.ProgressBar(len(image_name_list))
    pp.start()
    pp_idx = 0
    print ('Start loading raw images...')
    processed_image_set = set()
    for one_image_name in image_name_list:
        pp.update(pp_idx)
        pp_idx += 1
        if one_image_name in processed_image_set:
            continue
        processed_image_set.add(one_image_name)
        one_image_path = prefix_path + '/' + one_image_name
        image_instance_list.append(copy.deepcopy(Image.open(one_image_path)))
    assert len(image_instance_list) == len(image_name_list)
    pp.finish()
    print ('Raw image loaded!')

    batched_image_instance_list, batched_image_name_list = [], []
    image_num = len(image_instance_list)
    batch_num = image_num // batch_size
    s_idx, e_idx = 0, batch_size
    print ('Start loading images...')
    p = progressbar.ProgressBar(batch_num)
    p.start()
    for p_idx in range(batch_num):
        p.update(p_idx)
        one_batch_image_instance, one_batch_image_name = [], []
        for idx in range(s_idx, e_idx):
            one_batch_image_instance.append(image_instance_list[idx])
            one_batch_image_name.append(image_name_list[idx])
        assert len(one_batch_image_instance) == len(one_batch_image_name)
        s_idx += batch_size
        e_idx += batch_size
        if len(one_batch_image_instance) == 0:
            continue
        else:
            batched_image_instance_list.append(one_batch_image_instance)
            batched_image_name_list.append(one_batch_image_name)
    p.finish()
    return batched_image_instance_list, batched_image_name_list


def load_all_image_names(prefix_path):
    filename_list = listdir(prefix_path)
    image_name_list = []
    for filename in filename_list:
        image_name_list.append(filename)
    image_name_list = list(set(image_name_list))
    return image_name_list

def load_batch_image_names(prefix_path, batch_size):
    image_name_list = load_all_image_names(prefix_path)
    print ('Number of images is {}'.format(len(image_name_list)))
    batched_image_name_list = []
    image_num = len(image_name_list)
    batch_num = image_num // batch_size
    s_idx, e_idx = 0, batch_size
    print ('Start loading images...')
    p = progressbar.ProgressBar(batch_num)
    p.start()
    for p_idx in range(batch_num):
        p.update(p_idx)
        one_batch_image_name = []
        for idx in range(s_idx, e_idx):
            one_batch_image_name.append(image_name_list[idx])
        s_idx += batch_size
        e_idx += batch_size
        if len(one_batch_image_name) == 0:
            continue
        else:
            batched_image_name_list.append(one_batch_image_name)
    p.finish()
    return batched_image_name_list


def parse_images(prefix_path, batch_size, image_format='jpg'):
    image_name_list = load_image_names(prefix_path, image_format)
    print ('Number of images are {}'.format(len(image_name_list)))
    batched_image_instance_list, batched_image_name_list = \
    load_image_instances(prefix_path, image_name_list, batch_size)
    return batched_image_instance_list, batched_image_name_list

