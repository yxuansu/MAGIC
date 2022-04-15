CUDA_VISIBLE_DEVICES=1 python ../inference_baseline.py\
    --language_model_code_path ../language_model/\
    --language_model_name cambridgeltl/magic_mscoco\
    --test_path ../data/mscoco/mscoco_test.json\
    --decoding_method topk\
    --decoding_len 16\
    --top_k 40\
    --save_path_prefix ../inference_result/mscoco/baselines/\
    --save_name top_k_result_run_1.json

