## Details of How to Run the Inference Code

In this folder, we illustrate how to conduct inference with different methods.

****
### Catalogue:
* <a href='#magic_search'>1. Magic Search</a>
    * <a href='#in_domain_magic_search'>1.1. In Domain Experiment</a>
    * <a href='#cross_domain_magic_search'>1.2. Cross Domain Experiment</a>
* <a href='#contrastive_search'>2. Contrastive Search</a>
    * <a href='#in_domain_contrastive_search'>2.1. In Domain Experiment</a>
    * <a href='#cross_domain_contrastive_search'>2.2. Cross Domain Experiment</a>
* <a href='#topk_sampling'>3. Top-k Sampling</a>
    * <a href='#in_domain_topk_sampling'>3.1. In Domain Experiment</a>
    * <a href='#cross_domain_topk_sampling'>3.2. Cross Domain Experiment</a>
* <a href='#nucleus_sampling'>4. Nucleus Sampling</a>
    * <a href='#in_domain_nucleus_sampling'>4.1. In Domain Experiment</a>
    * <a href='#cross_domain_nucleus_sampling'>4.2. Cross Domain Experiment</a>

****

<span id='magic_search'/>

### 1. Magic Search:

<span id='in_domain_magic_search'/>

#### 1.1. In Domain Experiment:

To perform in domain experiment with magic search, please run the following command:
```yaml
chmod +x ./X_magic.sh
./X_magic.sh
```
Here, X is in ['mscoco', 'flickr30k'] and the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface (cambridgeltl/magic_mscoco or cambridgeltl/magic_flickr30k) 
* `--clip_path`: Where the code of CLIP locates.
* `--clip_name`: The CLIP model name on huggingface. 
* `--test_image_prefix_path`: The directory that stores the test set images. 
* `--test_path`: The file that stores the reference captions. 
* `--decoding_len`: The number of tokens to generate. 
* `--k`: The k in magic search. 
* `--alpha`: The alpha in magic search. 
* `--beta`: The beta in magic search. 
* `--save_path_prefix`: Where to save the inferenced result. 
* `--save_name`: The saved name of the inferenced result. 

**[Note]** For in domain experiments, the test set and the language model (defined by the argument of language_model_name) should come from the same domain.

<span id='cross_domain_magic_search'/>

#### 1.2. Cross Domain Experiment:

To perform cross domian experiment with magic search, please run the following command:
```yaml
chmod +x ./source_X_target_Y_magic.sh
./source_X_target_Y_magic.sh
```
Here, X is the source domain from ['mscoco', 'flickr30k'] and Y is the target domain from ['flickr30k', 'mscoco']. 

The arguments are the same as in section 1.1.

**[Note]** For cross domain experiments, the test set and the language model (defined by the argument of language_model_name) should come from **different** domains.


****

<span id='contrastive_search'/>

### 2. Contrastive Search:

<span id='in_domain_contrastive_search'/>

#### 2.1. In Domain Experiment:

To perform in domain experiment with contrastive search, please run the following command:
```yaml
chmod +x ./X_contrastive_baseline.sh
./X_contrastive_baseline.sh
```
Here, X is in ['mscoco', 'flickr30k'] and the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface (cambridgeltl/magic_mscoco or cambridgeltl/magic_flickr30k) 
* `--test_path`: The file that stores the reference captions. 
* `--decoding_method`: contrastive  
* `--decoding_len`: The number of tokens to generate. 
* `--k`: The k in contrastive search. 
* `--alpha`: The alpha in contrastive search. 
* `--save_path_prefix`: Where to save the inferenced result. 
* `--save_name`: The saved name of the inferenced result. 

**[Note]** For in domain experiments, the test set and the language model (defined by the argument of language_model_name) should come from the same domain.

<span id='cross_domain_contrastive_search'/>

#### 2.2. Cross Domain Experiment:

To perform cross domian experiment with contrastive search, please run the following command:
```yaml
chmod +x ./source_X_target_Y_contrastive_baseline.sh
./source_X_target_Y_contrastive_baseline.sh
```
Here, X is the source domain from ['mscoco', 'flickr30k'] and Y is the target domain from ['flickr30k', 'mscoco']. 

The arguments are the same as in section 2.1.

**[Note]** For cross domain experiments, the test set and the language model (defined by the argument of language_model_name) should come from **different** domains.


****

<span id='topk_sampling'/>

### 3. Top-k Sampling:

<span id='in_domain_topk_sampling'/>

#### 3.1. In Domain Experiment:

To perform in domain experiment with top-k sampling, please run the following command:
```yaml
chmod +x ./X_topk_baseline.sh
./X_topk_baseline.sh
```
Here, X is in ['mscoco', 'flickr30k'] and the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface (cambridgeltl/magic_mscoco or cambridgeltl/magic_flickr30k) 
* `--test_path`: The file that stores the reference captions. 
* `--decoding_method`: topk  
* `--decoding_len`: The number of tokens to generate. 
* `--top_k`: The k in top-k sampling. 
* `--save_path_prefix`: Where to save the inferenced result. 
* `--save_name`: The saved name of the inferenced result. 

**[Note]** For in domain experiments, the test set and the language model (defined by the argument of language_model_name) should come from the same domain.

<span id='cross_domain_topk_sampling'/>

#### 3.2. Cross Domain Experiment:

To perform cross domian experiment with top-k sampling, please run the following command:
```yaml
chmod +x ./source_X_target_Y_topk_baseline.sh
./source_X_target_Y_topk_baseline.sh
```
Here, X is the source domain from ['mscoco', 'flickr30k'] and Y is the target domain from ['flickr30k', 'mscoco']. 

The arguments are the same as in section 3.1.

**[Note]** For cross domain experiments, the test set and the language model (defined by the argument of language_model_name) should come from **different** domains.

****

<span id='nucleus_sampling'/>

### 4. Nucleus Sampling:

<span id='in_domain_nucleus_sampling'/>

#### 4.1. In Domain Experiment:

To perform in domain experiment with nucleus sampling, please run the following command:
```yaml
chmod +x ./X_nucleus_baseline.sh
./X_nucleus_baseline.sh
```
Here, X is in ['mscoco', 'flickr30k'] and the arguments are as follows:
* `--language_model_code_path`: Where the code of language model locates. 
* `--language_model_name`: The language model name on huggingface (cambridgeltl/magic_mscoco or cambridgeltl/magic_flickr30k) 
* `--test_path`: The file that stores the reference captions. 
* `--decoding_method`: nucleus
* `--decoding_len`: The number of tokens to generate. 
* `--nucleus_p`: The p in nucleus sampling. 
* `--save_path_prefix`: Where to save the inferenced result. 
* `--save_name`: The saved name of the inferenced result. 

**[Note]** For in domain experiments, the test set and the language model (defined by the argument of language_model_name) should come from the same domain.

<span id='cross_domain_nucleus_sampling'/>

#### 4.2. Cross Domain Experiment:

To perform cross domian experiment with nucleus sampling, please run the following command:
```yaml
chmod +x ./source_X_target_Y_nucleus_baseline.sh
./source_X_target_Y_nucleus_baseline.sh
```
Here, X is the source domain from ['mscoco', 'flickr30k'] and Y is the target domain from ['flickr30k', 'mscoco']. 

The arguments are the same as in section 4.1.

**[Note]** For cross domain experiments, the test set and the language model (defined by the argument of language_model_name) should come from **different** domains.

