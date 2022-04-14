CUDA_VISIBLE_DEVICES=1 python ../inference_baseline.py\
    --language_model_code_path ../language_model/\
    --language_model_name cambridgeltl/magic_mscoco\
    --test_path ../data/mscoco/mscoco_test.json\
    --decoding_method contrastive\
    --decoding_len 16\
    --k 45\
    --alpha 0.1\
    --save_path_prefix ../inference_result/mscoco/baselines/\
    --save_name contrastive_result.json

