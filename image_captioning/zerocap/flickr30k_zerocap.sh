#!/bin/bash

# lm_model:
#  1. cambridgeltl/magic_mscoco
#  2. cambridgeltl/magic_flickr30k
CUDA_VISIBLE_DEVICES=1 python run.py \
    --beam_size 1 \
    --target_seq_length 16 \
    --reset_context_delta \
    --lm_model cambridgeltl/magic_flickr30k \
    --test_image_prefix_path ../data/flickr30k/test_images \
    --test_path ../data/flickr30k/flickr30k_test.json \
    --save_path_prefix ../inference_result/flickr30k/baselines/ \
    --save_name zerocap_result.json
