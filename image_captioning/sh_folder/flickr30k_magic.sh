CUDA_VISIBLE_DEVICES=0 python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name cambridgeltl/magic_flickr30k\
    --clip_path ../clip/\
    --clip_name openai/clip-vit-base-patch32\
    --test_image_prefix_path ../data/flickr30k/test_images/\
    --test_path ../data/flickr30k/flickr30k_test.json\
    --decoding_len 20\
    --k 25\
    --alpha 0.1\
    --beta 2.0\
    --save_path_prefix ../inference_result/flickr30k/\
    --save_name magic_result.json