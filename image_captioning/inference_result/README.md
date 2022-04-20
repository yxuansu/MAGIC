## Inference Results
This directory contains the results of all evaluated methods for the image captioning task.

### 1. Directory Structure:
> ****  The structure of the directory looks like:

    .
    ├── ./mscoco/  # results on MSCOCO benchmark               
        ├── magic_result.json # The result of our magic approach
        ├── baselines/ # The directory that contains all baseline results
            ├── contrastive_result.json # contrastive search 
            ├── top_k_result_run_1.json # first run of top-k sampling 
            ├── top_k_result_run_2.json # second run of top-k sampling 
            ├── top_k_result_run_3.json # third run of top-k sampling 
            ├── nucleus_result_run_1.json # first run of nucleus sampling 
            ├── nucleus_result_run_2.json # second run of nucleus sampling 
            ├── nucleus_result_run_3.json # third run of nucleus sampling 
            ├── mscoco_in_domain_clipretrieval.json # CLIPRe 
            └── zerocap_result.json # ZeroCap   
    ├── ./flickr30k/  # results on Flickr30k benchmark
        ├── magic_result.json # The result of our magic approach
        ├── baselines/ # The directory that contains all baseline results
            ├── contrastive_result.json # contrastive search 
            ├── top_k_result_run_1.json # first run of top-k sampling 
            ├── top_k_result_run_2.json # second run of top-k sampling 
            ├── top_k_result_run_3.json # third run of top-k sampling 
            ├── nucleus_result_run_1.json # first run of nucleus sampling 
            ├── nucleus_result_run_2.json # second run of nucleus sampling 
            ├── nucleus_result_run_3.json # third run of nucleus sampling 
            ├── flickr30k_in_domain_clipretrieval.json # CLIPRe 
            └── zerocap_result.json # ZeroCap
    ├── ./flickr30k_model_to_mscoco/ # results of cross domain image captioning on MSCOCO benchmark
        ├── magic_result.json # The result of our magic approach 
        ├── contrastive_result.json # contrastive search 
        ├── top_k_result_run_1.json # first run of top-k sampling 
        ├── top_k_result_run_2.json # second run of top-k sampling
        ├── top_k_result_run_3.json # third run of top-k sampling 
        ├── nucleus_result_run_1.json # first run of nucleus sampling 
        ├── nucleus_result_run_2.json # second run of nucleus sampling
        ├── nucleus_result_run_3.json # third run of nucleus sampling 
        └── source_flickr30k_target_mscoco_clip_retrieval.json # CLIPRe 
    ├── ./mscoco_model_to_flickr30k/ # results of cross domain image captioning on Flickr30k benchmark
        ├── magic_result.json # The result of our magic approach 
        ├── contrastive_result.json # contrastive search 
        ├── top_k_result_run_1.json # first run of top-k sampling 
        ├── top_k_result_run_2.json # second run of top-k sampling
        ├── top_k_result_run_3.json # third run of top-k sampling 
        ├── nucleus_result_run_1.json # first run of nucleus sampling 
        ├── nucleus_result_run_2.json # second run of nucleus sampling
        ├── nucleus_result_run_3.json # third run of nucleus sampling 
        └── source_mscoco_target_flickr30k_clip_retrieval.json # CLIPRe 

### 2. Data Format of the Inferenced Result:

The generated file is a list of dictionary, where the data format of each dictionary is:
```yaml
{  
   "split": Indicating which split (train, val, or test) the data instance belongs to.
   "image_name": The name of the corresponding image.
   "captions": A list of captions that the data instance contains.
   "prediction": The predicted result of the model.
}
```
