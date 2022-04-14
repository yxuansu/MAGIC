CUDA_VISIBLE_DEVICES=0 python build_text_index.py\
    --clip_name openai/clip-vit-base-patch32\
    --text_file_path ../data/mscoco/mscoco_train.json\
    --save_index_prefix ./mscoco_index/\
    --save_index_name index_matrix.txt\
    --save_mapping_dict_name text_mapping.json\
    --batch_size 128