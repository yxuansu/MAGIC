CUDA_VISIBLE_DEVICES=1 python ../inference_baseline.py\
    --language_model_code_path ../language_model/\
    --language_model_name cambridgeltl/magic_flickr30k\
    --test_path ../data/flickr30k/flickr30k_test.json\
    --decoding_method topk\
    --decoding_len 20\
    --top_k 40\
    --save_path_prefix ../inference_result/flickr30k/baselines/\
    --save_name top_k_result_run_1.json

