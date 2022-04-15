CUDA_VISIBLE_DEVICES=0 python ../inference_baseline.py\
    --language_model_code_path ../language_model/\
    --language_model_name cambridgeltl/magic_mscoco\
    --test_path ../data/flickr30k/flickr30k_test.json\
    --decoding_method nucleus\
    --decoding_len 16\
    --nucleus_p 0.95\
    --save_path_prefix ../inference_result/mscoco_model_to_flickr30k/\
    --save_name nucleus_result_run_1.json

