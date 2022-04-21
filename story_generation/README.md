## Visually Grounded Story Generaton
In this directory, we illustrate the details of our experiments on the task of visually grounded story generation. 

****
### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#language_model_training'>2. Language Model Training</a>
* <a href='#inference_with_magic'>3. Perform Inference with Magic</a>
* <a href='#inference_with_baseline'>4. Perform Inference with Baseline Methods</a>
    * <a href='#contrastive_search'>4.1. Contrastive Search</a>
    * <a href='#greedy_search'>4.2. Greedy Search</a>
    * <a href='#beam_search'>4.3. Beam Search</a>
    * <a href='#topk_sampling'>4.4. Top-k Sampling</a>
    * <a href='#nucleues_sampling'>4.5. Nucleus Sampling</a>
    * <a href='#typical_sampling'>4.6. Typical Sampling</a>

****

<span id='data_preparation'/>

### 1. Data Preparation:
To prepare the data and image index for the task, please follow instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/data).


****

<span id='language_model_training'/>

### 2. Language Model Training:
To train the language model on the ROCStories benchmark, please follow the instructions [[here]](https://github.com/yxuansu/SimCTG/tree/main/story_generation#2-open-ended-story-generation-on-rocstories-benchmark).

****

<span id='inference_with_magic'/>

### 3. Perform Inference with Magic:
To perform inference with our magic approach, please refer to details [[here]](https://github.com/yxuansu/MAGIC#62-example-usage-of-magic-search) and [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/sh_folder#1-magic-search).

****

<span id='inference_with_baseline'/>

### 4. Perform Inference with Baseline Methods:

<span id='contrastive_search'/>

#### 4.1. Contrastive Search:
To perform inference with the contrastive search baseline, please refer to details [[here]](hhttps://github.com/yxuansu/MAGIC/tree/main/story_generation/sh_folder#2-contrastive-search).


<span id='greedy_search'/>

#### 4.2. Greedy Search:
To perform inference with the greedy search baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/sh_folder#3-greedy-search).

<span id='beam_search'/>

#### 4.3. Beam Search:
To perform inference with the beam search baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/sh_folder#4-beam-search).


<span id='topk_sampling'/>

#### 4.4. Top-k Sampling:
To perform inference with the top-k sampling baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/sh_folder#5-top-k-sampling).

<span id='nucleues_sampling'/>

#### 4.5. Nucleus Sampling:
To perform inference with the nucleus sampling baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/sh_folder#6-nucleus-sampling).

<span id='typical_sampling'/>

#### 4.6. Typical Sampling:
To perform inference with the typical sampling baseline, please refer to details [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/sh_folder#7-typical-sampling).
