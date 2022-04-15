CUDA_VISIBLE_DEVICES=2 python ../inference_baseline.py\
    --language_model_code_path ../language_model/\
    --language_model_name cambridgeltl/magic_flickr30k\
    --test_path ../data/mscoco/mscoco_test.json\
    --decoding_method contrastive\
    --decoding_len 20\
    --k 25\
    --alpha 0.1\
    --save_path_prefix ../inference_result/flickr30k_model_to_mscoco/\
    --save_name contrastive_result.json

