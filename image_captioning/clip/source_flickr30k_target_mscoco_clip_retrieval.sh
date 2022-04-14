CUDA_VISIBLE_DEVICES=1 python clipretrieval.py\
    --clip_name openai/clip-vit-base-patch32\
    --test_image_prefix_path ../data/mscoco/test_images/\
    --test_path ../data/mscoco/mscoco_test.json\
    --index_matrix_path ./flickr30k_index/index_matrix.txt\
    --mapping_dict_path ./flickr30k_index/text_mapping.json\
    --save_path_prefix ../inference_result/flickr30k_model_to_mscoco/\
    --save_name source_flickr30k_target_mscoco_clip_retrieval.json