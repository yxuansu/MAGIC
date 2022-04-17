## Zero-Shot Image Captioning
In this directory, we illustrate the details of our experiments on the task of zero-shot image captioning. 

****
### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#unsupervised_domain_adaptation'>2. Unsupervised Domain Adaptation</a>
    * <a href='#mscoco_adaptation'>2.1. MSCOCO</a>
    * <a href='#flickr30k_adaptation'>2.2. Flickr30k</a>
* <a href='#inference_with_magic'>3. Perform Inference with Magic</a>
* <a href='#inference_with_baseline'>4. Perform Inference with Baseline Methods</a>
    * <a href='#topk_sampling'>4.1. Top-k Sampling</a>
    * <a href='#nucleues_sampling'>4.2. Nucleus Sampling</a>
    * <a href='#contrastive_search'>4.3. Contrastive Search</a>
    * <a href='#clipre'>4.4. CLIPRe</a>
    * <a href='#zerocap'>4.5. ZeroCap</a>


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
