CUDA_VISIBLE_DEVICES=1 python clipretrieval.py\
    --clip_name openai/clip-vit-base-patch32\
    --test_image_prefix_path ../data/flickr30k/test_images/\
    --test_path ../data/flickr30k/flickr30k_test.json\
    --index_matrix_path ./flickr30k_index/index_matrix.txt\
    --mapping_dict_path ./flickr30k_index/text_mapping.json\
    --save_path_prefix ../inference_result/flickr30k/baselines/\
    --save_name flickr30k_in_domain_clipretrieval.json