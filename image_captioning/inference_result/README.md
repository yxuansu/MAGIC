## Inference Results
This directory contains the results of all evaluated methods for the image captioning task.

> ****  The structure of the directory looks like:

    .
    ├── ./mscoco/  # results on MSCOCO benchmark               
        ├── magic_result.json # The results of our magic approach
        ├── baselines/ # The directory that contains all baseline results
            ├── contrastive_result.json # contrastive search baseline
            ├── top_k_result_run_1.json # first run of top-k sampling baseline
            ├── top_k_result_run_2.json # second run of top-k sampling baseline
            ├── top_k_result_run_3.json # third run of top-k sampling baseline
            ├── nucleus_result_run_1.json # first run of nucleus sampling baseline
            ├── nucleus_result_run_2.json # second run of nucleus sampling baseline
            ├── nucleus_result_run_3.json # third run of nucleus sampling baseline
            ├── mscoco_in_domain_clipretrieval.json # CLIPRe baseline
            └── zerocap_result # ZeroCap baseline
    ├── ./flickr30k/  # results on Flickr30k benchmark
        ├── magic_result.json # The results of our magic approach
        ├── baselines/ # The directory that contains all baseline results
            ├── contrastive_result.json # contrastive search baseline
            ├── top_k_result_run_1.json # first run of top-k sampling baseline
            ├── top_k_result_run_2.json # second run of top-k sampling baseline
            ├── top_k_result_run_3.json # third run of top-k sampling baseline
            ├── nucleus_result_run_1.json # first run of nucleus sampling baseline
            ├── nucleus_result_run_2.json # second run of nucleus sampling baseline
            ├── nucleus_result_run_3.json # third run of nucleus sampling baseline
            ├── flickr30k_in_domain_clipretrieval.json # CLIPRe baseline
            └── zerocap_result # ZeroCap baseline
    ├── ./flickr30k_model_to_mscoco/ # results of cross domain image captioning on MSCOCO benchmark
        ├──
        ├──
        ├──
        ├──
        ├──
        ├──
        ├──
        ├──
        ├──
        └──  # ZeroCap baseline
    ├── ./mscoco_model_to_flickr30k/ # results of cross domain image captioning on Flickr30k benchmark
