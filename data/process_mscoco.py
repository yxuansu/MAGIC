import json
def parse_split(in_f):
    '''
        each item in the split has the following format
        {
            'split': train, val, or test
            'image_name' xxx.jpg,
            'file_path': ...,
            'captions': [sentence_1,
                        sentence_2,
                        ...,
                        sentence_5]   
        }
    '''
    with open(in_f) as f:
        data = json.load(f)
        
    train_split, val_split, test_split = [], [], []
    for item in data['images']:
        filepath = item['filepath']
        image_name = item['filename']
        split = item['split']
        caption_list = []
        for sen in item['sentences']:
            one_caption = sen['raw'].strip()
            caption_list.append(one_caption)
        one_dict = {
            'split':split,
            'image_name':image_name,
            'file_path':filepath,
            'captions':caption_list
        }
        if split == 'train':
            train_split.append(one_dict)
        elif split == 'val':
            val_split.append(one_dict)
        elif split == 'test':
            test_split.append(one_dict)
        else:
            pass
    return train_split, val_split, test_split

import shutil
def copy_images(test_split, raw_image_prefix_path, copy_directory_prefix):
    import os
    if os.path.exists(copy_directory_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(copy_directory_prefix, exist_ok=True)

    number_of_copied_images = 0
    for item in test_split:
        split_path = item['file_path']
        image_name = item['image_name']
        image_full_path = raw_image_prefix_path + '/' + split_path + '/' + image_name
        copy_path = copy_directory_prefix + '/' + image_name
        shutil.copyfile(image_full_path, copy_path)
        number_of_copied_images += 1
    print ('Number of test images is {}'.format(number_of_copied_images))

if __name__ == '__main__':
    import os
    save_path = r'./mscoco/'
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    import json
    in_f = r'./raw_data/caption_datasets/dataset_coco.json'
    train_split, val_split, test_split = parse_split(in_f)
    print ('Number of train instance {}, val instances {}, and test instances {}'.format(len(train_split),
        len(val_split), len(test_split)))

    train_save_path = save_path + '/' + r'mscoco_train.json'
    with open(train_save_path, 'w') as outfile:
        json.dump(train_split, outfile, indent=4)

    val_save_path = save_path + '/' + r'mscoco_val.json'
    with open(val_save_path, 'w') as outfile:
        json.dump(val_split, outfile, indent=4)

    test_save_path = save_path + '/' + r'mscoco_test.json'
    with open(test_save_path, 'w') as outfile:
        json.dump(test_split, outfile, indent=4)

    raw_image_prefix_path = r'./raw_images/mscoco/'
    copy_directory_prefix = r'./mscoco/test_images/'
    copy_images(test_split, raw_image_prefix_path, copy_directory_prefix)
