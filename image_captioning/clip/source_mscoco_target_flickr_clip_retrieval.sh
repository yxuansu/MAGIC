CUDA_VISIBLE_DEVICES=1 python clipretrieval.py\
    --clip_name openai/clip-vit-base-patch32\
    --test_image_prefix_path ../data/flickr30k/test_images/\
    --test_path ../data/flickr30k/flickr30k_test.json\
    --index_matrix_path ./mscoco_index/index_matrix.txt\
    --mapping_dict_path ./mscoco_index/text_mapping.json\
    --save_path_prefix ../inference_result/mscoco_model_to_flickr30k/\
    --save_name source_mscoco_target_flickr30k_clip_retrieval.json