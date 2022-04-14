CUDA_VISIBLE_DEVICES=0 python clipretrieval.py\
    --clip_name openai/clip-vit-base-patch32\
    --test_image_prefix_path ../data/mscoco/test_images/\
    --test_path ../data/mscoco/mscoco_test.json\
    --index_matrix_path ./mscoco_index/index_matrix.txt\
    --mapping_dict_path ./mscoco_index/text_mapping.json\
    --save_path_prefix ../inference_result/mscoco/baselines/\
    --save_name mscoco_in_domain_clipretrieval.json