#!/bin/bash

# lm_model:
#  1. cambridgeltl/magic_mscoco
#  2. cambridgeltl/magic_flickr30k
CUDA_VISIBLE_DEVICES=6 python run.py \
    --beam_size 1 \
    --target_seq_length 16 \
    --reset_context_delta \
    --lm_model cambridgeltl/magic_flickr30k \
    --test_image_prefix_path ../data/mscoco/test_images \
    --test_path ../data/mscoco/mscoco_test.json \
    --save_path_prefix ../inference_result/flickr30k_model_to_mscoco/ \
    --save_name zerocap_result.json
