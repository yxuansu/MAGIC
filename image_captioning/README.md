## Zero-Shot Image Captioning
In this directory, we illustrate the details of our experiments on the task of zero-shot image captioning. 

> ****  The structure of the directory looks like:

    .
    ├──
        ├── ./data/  # Contains details of how to prepare the benchmark data
        ├── ./language_model/ # Contains details of the unsupervised domain adaptation of language and 
                                the details of different generation methods
        ├── ./clip/  # Contains details of the CLIP model and how to perform the CLIPRe baseline
        ├── ./evaluation/ # Contains details on how to perform evaluation on the inferenced results
        ├── ./sh_folder/ # Contains details on how to perform inference with different methods
        ├── ./zerocap/ # Contains details of the ZeroCap baseline
        └── ./inference_result/ # Contains the inferenced results of all evaluated methods

**[Note]** To ensure the reproductivity of our work, [in this folder](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/inference_result), we provide the inferenced results of all evaluated methods.

****
### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#unsupervised_domain_adaptation'>2. Unsupervised Domain Adaptation</a>
    * <a href='#mscoco_adaptation'>2.1. MSCOCO</a>
    * <a href='#flickr30k_adaptation'>2.2. Flickr30k</a>
    * <a href='#huggingface_models'>2.3. Huggingface Models</a>
* <a href='#inference_with_magic'>3. Perform Inference with Magic</a>
* <a href='#inference_with_baseline'>4. Perform Inference with Baseline Methods</a>
    * <a href='#topk_sampling'>4.1. Top-k Sampling</a>
    * <a href='#nucleues_sampling'>4.2. Nucleus Sampling</a>
    * <a href='#contrastive_search'>4.3. Contrastive Search</a>
    * <a href='#clipre'>4.4. CLIPRe</a>
    * <a href='#zerocap'>4.5. ZeroCap</a>
* <a href='#evaluation'>5. Perform Evaluation</a> 


****

<span id='data_preparation'/>

### 1. Data Preparation:
To prepare the data for MSCOCO and Flickr30k benchmarks, please follow instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/data).


****

<span id='unsupervised_domain_adaptation'/>

### 2. Unsupervised Domain Adaptation:

<span id='mscoco_adaptation'/>

#### 2.1. MSCOCO:
To perform unsupervised domain adaptation on MSCOCO domain, please follow the instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/language_model#12unsupervised-domain-adaptation-on-mscoco). 


<span id='flickr30k_adaptation'/>

#### 2.2. Flickr30k:
To perform unsupervised domain adaptation on Flickr30k domain, please follow the instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/language_model#22-unsupervised-domain-adaptation-on-flickr30k). 

<span id='huggingface_models'/>

#### 2.3. Huggingface Models:
We provide our language models adapted on MSCOCO and Flickr30k for an easy usage and the reproductivity of our results.

|Model Name|Training Corpus|Model Size|Model Address|
|:-------------:|:-------------:|:-------------:|:-------------:|
|cambridgeltl/magic_mscoco|MSCOCO|117M|[[link]](https://huggingface.co/cambridgeltl/magic_mscoco/)|
|cambridgeltl/magic_flickr30k|Flickr30k|117M|[[link]](https://huggingface.co/cambridgeltl/magic_flickr30k/)|



****

<span id='inference_with_magic'/>

### 3. Perform Inference with Magic:
To perform inference with our magic approach, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/sh_folder#1-magic-search).

****

<span id='inference_with_baseline'/>

### 4. Perform Inference with Baseline Methods:

<span id='topk_sampling'/>

#### 4.1. Top-k Sampling:
To perform inference with the top-k sampling baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/sh_folder#3-top-k-sampling) and [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/language_model#32-top-k-sampling-).

<span id='nucleues_sampling'/>

#### 4.2. Nucleus Sampling:
To perform inference with the nucleus sampling baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/sh_folder#4-nucleus-sampling) and [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/language_model#33-nucleus-sampling-).

<span id='contrastive_search'/>

#### 4.3. Contrastive Search:
To perform inference with the contrastive search baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/sh_folder#2-contrastive-search) and [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/language_model#31-contrastive-search-).

<span id='clipre'/>

#### 4.4. CLIPRe:
To perform inference with the CLIPRe baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/clip#clip).

<span id='zerocap'/>

#### 4.5. ZeroCap:
To perform inference with the ZeroCap baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/zerocap).


****

<span id='evaluation'/>

### 5. Perform Evaluation:
To obtain the numerical evaluation on the inferenced results of the model, please follow the instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/evaluation).

