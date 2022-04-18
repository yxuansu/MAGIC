CUDA_VISIBLE_DEVICES=0 python build_index.py\
    --clip_name openai/clip-vit-base-patch32\
    --image_file_prefix_path ../data/image_index/images/\
    --image_format jpg\
    --save_index_prefix ../data/image_index/images_index_data/\
    --save_index_name index_matrix.txt\
    --save_image_name_dict mapping_dict.json\
    --batch_size 256
