CUDA_VISIBLE_DEVICES=2 python ../inference_baseline.py\
    --language_model_code_path ../language_model/\
    --language_model_name cambridgeltl/simctg_rocstories\
    --test_path ../data/rocstories_test.txt\
    --num_of_inference_instances 1500\
    --decoding_len 100\
    --number_of_instance_to_generate_per_method 1\
    --decoding_method beam\
    --beam_width 10\
    --save_path_prefix ../inference_result