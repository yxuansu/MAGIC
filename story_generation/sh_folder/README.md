## Details of How to Run the Inference Code

In this folder, we illustrate how to conduct inference with different methods.

****
### Catalogue:
* <a href='#magic_search'>1. Magic Search</a>
* <a href='#contrastive_search'>2. Contrastive Search</a>
* <a href='#greedy_search'>3. Greedy Search</a>
* <a href='#beam_search'>4. Beam Search</a>
* <a href='#topk_sampling'>5. Top-k Sampling</a>
* <a href='#nucleus_sampling'>6. Nucleus Sampling</a>
* <a href='#typical_sampling'>7. Typical Sampling</a>

****

<span id='magic_search'/>

### 1. Magic Search:

To perform inference with magic search, please run the following command:
```yaml
chmod +x ./magic.sh
./magic.sh
```
Here, the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface.
* `--image_index_code_path`: Where the code of image index locates. 
* `--clip_path`: Where the code of CLIP locates.
* `--clip_name`: The CLIP model name on huggingface. 
* `--image_index_matrix_path`: The path of the image index representation matrix file.
* `--image_mapping_dict_path`: The path of the image index mapping dictionary file.
* `--image_folder_prefix_path`: The path of the raw images from which we perform the image retrieval.
* `--test_path`: The file that stores the test set. 
* `--num_of_inference_instances`: How many instances to perform inference.
* `--number_of_instance_to_generate_per_method`: How many results we generate per instance.
* `--decoding_len`: The number of tokens to generate. 
* `--k`: The k in magic search. 
* `--alpha`: The alpha in magic search. 
* `--beta`: The beta in magic search. 
* `--save_path_prefix`: Where to save the inferenced result. 

**[Note]** We provide our inferenced result with magic search [[here]](https://github.com/yxuansu/MAGIC/blob/main/story_generation/inference_result/).

****

<span id='contrastive_search'/>

### 2. Contrastive Search:

To perform inference with contrastive search, please run the following command:
```yaml
chmod +x ./contrastive_search.sh
./contrastive_search.sh
```
Here, the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface (cambridgeltl/magic_mscoco or cambridgeltl/magic_flickr30k) 
* `--test_path`: The file that stores the reference captions. 
* `--num_of_inference_instances`: How many instances to perform inference.
* `--number_of_instance_to_generate_per_method`: How many results we generate per instance.
* `--decoding_len`: The number of tokens to generate. 
* `--decoding_method`: contrastive  
* `--k`: The k in contrastive search. 
* `--alpha`: The alpha in contrastive search. 
* `--save_path_prefix`: Where to save the inferenced result. 

**[Note]** We provide our inferenced result with contrastive search [[here]](https://github.com/yxuansu/MAGIC/blob/main/story_generation/inference_result/).


****

<span id='greedy_search'/>

### 3. Greedy Search:
To perform inference with greedy search, please run the following command:
```yaml
chmod +x ./greedy_search.sh
./greedy_search.sh
```
Here, the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface (cambridgeltl/magic_mscoco or cambridgeltl/magic_flickr30k) 
* `--test_path`: The file that stores the reference captions. 
* `--num_of_inference_instances`: How many instances to perform inference.
* `--number_of_instance_to_generate_per_method`: How many results we generate per instance.
* `--decoding_len`: The number of tokens to generate. 
* `--decoding_method`: greedy 
* `--save_path_prefix`: Where to save the inferenced result. 


****

<span id='beam_search'/>

### 4. Beam Search:
To perform inference with beam search, please run the following command:
```yaml
chmod +x ./beam_search.sh
./beam_search.sh
```
Here, the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface (cambridgeltl/magic_mscoco or cambridgeltl/magic_flickr30k) 
* `--test_path`: The file that stores the reference captions. 
* `--num_of_inference_instances`: How many instances to perform inference.
* `--number_of_instance_to_generate_per_method`: How many results we generate per instance.
* `--decoding_len`: The number of tokens to generate. 
* `--decoding_method`: beam
* `--beam_width`: The beam width for beam search.
* `--save_path_prefix`: Where to save the inferenced result. 



****

<span id='topk_sampling'/>

### 5. Top-k Sampling:
To perform inference with top-k sampling, please run the following command:
```yaml
chmod +x ./top_k_sampling.sh
./top_k_sampling.sh
```
Here, the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface (cambridgeltl/magic_mscoco or cambridgeltl/magic_flickr30k) 
* `--test_path`: The file that stores the reference captions. 
* `--num_of_inference_instances`: How many instances to perform inference.
* `--number_of_instance_to_generate_per_method`: How many results we generate per instance.
* `--decoding_len`: The number of tokens to generate. 
* `--decoding_method`: top-k
* `--top_k`: The k in top-k sampling.
* `--save_path_prefix`: Where to save the inferenced result. 


****

<span id='nucleus_sampling'/>

### 6. Nucleus Sampling:
To perform inference  with nucleus sampling, please run the following command:
```yaml
chmod +x ./nucleus_sampling.sh
./nucleus_sampling.sh
```
Here, the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface (cambridgeltl/magic_mscoco or cambridgeltl/magic_flickr30k) 
* `--test_path`: The file that stores the reference captions. 
* `--num_of_inference_instances`: How many instances to perform inference.
* `--number_of_instance_to_generate_per_method`: How many results we generate per instance.
* `--decoding_len`: The number of tokens to generate. 
* `--decoding_method`: nucleus
* `--nucleus_p`: The p in nucleus sampling
* `--save_path_prefix`: Where to save the inferenced result. 


