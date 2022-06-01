## Language Models Can See: Plugging Visual Controls in Text Generation
**Authors**: Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier


This repository contains code, models, and other related resources of our paper [[Language Models Can See:
Plugging Visual Controls in Text Generation]](https://arxiv.org/abs/2205.02655).

:star: If you are also interested in open-ended text generation and would like to see more details of our contrastive search decoding method, please refer to our SimCTG [[paper]](https://arxiv.org/abs/2202.06417) and [[repo]](https://github.com/yxuansu/SimCTG). 

:star: [Replicate](https://replicate.com/home) has provided a great web [[demo]](https://replicate.com/yxuansu/magic/examples) of MAGIC that is super easy to use and to interact with. Check it out!

****

![MAGIC](/demo.gif)

****
## Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#news'>2. News</a>
* <a href='#citation'>3. Citation</a>
* <a href='#environment_setup'>4. Environment Setup</a>
* <a href='#image_captioning'>5. Zero-Shot Image Captioning</a>
    * <a href='#image_captioning_experiment'>5.1. Implementation of Experiments</a>
    * <a href='#image_captioning_magic_search'>5.2. Example Usage of Magic Search</a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NDVkKpanbsaUwecHoRp_2kIpMztOFW25?usp=sharing)
         * <a href='#image_captioning_language_model'>5.2.1. Load Language Model</a>
         * <a href='#image_captioning_CLIP'>5.2.2. Load CLIP</a>
         * <a href='#image_captioning_start_token'>5.2.3. Prepare Start Token</a>
         * <a href='#image_captioning_load_image'>5.2.4. Load Image</a>
         * <a href='#image_captioning_magic_search_result'>5.2.5. Zero-Shot Image Captioning with Magic Search</a>
         * <a href='#image_captioning_reproduce_result'>5.2.6. Reproduce Our Results in the Paper</a>
* <a href='#story_generation'>6. Visually Grounded Story Generation</a>
    * <a href='#story_generation_experiment'>6.1. Implementation of Experiments</a>
    * <a href='#story_generation_magic_search'>6.2. Example Usage of Magic Search</a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19lyyMXDRNr-Op8vwUOiRmbhMxI_s3rwW?usp=sharing)
         * <a href='#story_generation_language_model'>6.2.1. Load Language Model</a>
         * <a href='#story_generation_CLIP'>6.2.2. Load CLIP</a>
         * <a href='#story_generation_get_image'>6.3.2. Get the Related Image</a>
              * <a href='#story_generation_get_image_from_index'>6.3.2.1. Retrieve from Image Index</a>
              * <a href='#story_generation_get_image_from_example'>6.3.2.2. Directly Load Image</a>
         * <a href='#story_generation_magic_search_result'>6.3.3. Visually Grounded Story Generation with Magic Search</a>
         * <a href='#story_generation_reproduce_result'>6.3.4. Reproduce Our Results in the Paper</a>
* <a href='#contact'>7. Contact</a>
* <a href='#magic_elsewhere'>8. MAGIC Elsewhere</a>

****

<span id='introduction'/>

### 1. Introduction:
Generative language models (LMs) such as GPT-2/3 can be prompted to generate text with remarkable quality. While they are designed for text-prompted generation, it remains an open question how the generation process could be guided by modalities beyond text such as images. In this work, we propose a training-free framework, called MAGIC (i<ins>**MA**</ins>ge-<ins>**G**</ins>uided text generat<ins>**I**</ins>on with <ins>**C**</ins>LIP), for plugging in visual controls in the generation process and enabling LMs to perform multimodal tasks (e.g., image captioning) in a zero-shot manner. MAGIC is a simple yet efficient plug-and-play framework, which directly combines an off-the-shelf LM (i.e., GPT-2) and an image-text matching model (i.e., CLIP) for image-grounded text generation. During decoding, MAGIC influences the generation of the LM by introducing a CLIP-induced score, called **_magic score_**, which regularizes the generated result to be semantically related to a given image while being coherent to the previously generated context. Notably, the proposed decoding scheme does not involve any gradient update operation, therefore being computationally efficient. On the challenging task of zero-shot image captioning, MAGIC outperforms the state-of-the-art method by notable margins with a nearly 27 times decoding speedup. MAGIC is a flexible framework and is theoretically compatible with any text generation tasks that incorporate image grounding. In the experiments, we showcase that it is also capable of performing visually grounded story generation given both an image and a text prompt.
****

<span id='news'/>

### 2. News:
* [2022/05/06] MAGIC is publicly released!
****

<span id='citation'/>

### 3. Citation:
If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```bibtex
@article{su2022language,
  title={Language Models Can See: Plugging Visual Controls in Text Generation},
  author={Su, Yixuan and Lan, Tian and Liu, Yahui and Liu, Fangyu and Yogatama, Dani and Wang, Yan and Kong, Lingpeng and Collier, Nigel},
  journal={arXiv preprint arXiv:2205.02655},
  year={2022}
}

@article{su2022contrastive,
  title={A Contrastive Framework for Neural Text Generation},
  author={Su, Yixuan and Lan, Tian and Wang, Yan and Yogatama, Dani and Kong, Lingpeng and Collier, Nigel},
  journal={arXiv preprint arXiv:2202.06417},
  year={2022}
}
```

****

<span id='environment_setup'/>

### 4. Environment Setup:
```yaml
python version: 3.8
pip3 install -r requirements.txt
```

****

<span id='image_captioning'/>

### 5. Zero-Shot Image Captioning:

<span id='image_captioning_experiment'/>

#### 5.1. Implementation of Experiments: 
To ensure the reproductity of our work, we provide all related resources to implement our experiments on the task of zero-shot image captioning. Please refer more details [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning). 

<span id='image_captioning_magic_search'/>

#### 5.2. Example Usage of Magic Search: 
In the following, we illustrate how to perform zero-shot image captioning with magic search. Specifically, we show how to generate the results as shown in our case study in the paper.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NDVkKpanbsaUwecHoRp_2kIpMztOFW25?usp=sharing)

<span id='image_captioning_language_model'/>

##### 5.2.1. Load Language Model:
We first load the language model as:
```python
import sys
sys.path.append(r'./image_captioning/language_model/')
from simctg import SimCTG
language_model_name = r'cambridgeltl/magic_mscoco'
sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
generation_model = SimCTG(language_model_name, sos_token, pad_token)
generation_model.eval()
```

<span id='image_captioning_CLIP'/>

##### 5.2.2. Load CLIP: 
Then, we load the CLIP model as:
```python
import sys
sys.path.append(r'./image_captioning/clip/')
from clip import CLIP
model_name = "openai/clip-vit-base-patch32"
clip = CLIP(model_name)
clip.eval()
```

<span id='image_captioning_start_token'/>

##### 5.2.3. Prepare Start Token: 
Note that, the language model always starts generation with a start of sentence token. Here, we prepare the input ids of the start token.
```python
import torch
sos_token = r'<-start_of_text->'
start_token = generation_model.tokenizer.tokenize(sos_token)
start_token_id = generation_model.tokenizer.convert_tokens_to_ids(start_token)
input_ids = torch.LongTensor(start_token_id).view(1,-1)
```

<span id='image_captioning_load_image'/>

##### 5.2.4. Load Image: 
To generate the caption of a random image, we need to load the image as:
```python
from PIL import Image             # to load images
from IPython.display import display # to display images
image_name_list = ['COCO_val2014_000000336777.jpg', 'COCO_val2014_000000182784.jpg', 'COCO_val2014_000000299319.jpg', 'COCO_val2014_000000516750.jpg',
                   'COCO_val2014_000000207151.jpg', 'COCO_val2014_000000078707.jpg', 'COCO_val2014_000000027440.jpg', 'COCO_val2014_000000033645.jpg',
                   'COCO_val2014_000000348905.jpg', 'COCO_val2014_000000545385.jpg', 'COCO_val2014_000000210032.jpg', 'COCO_val2014_000000577526.jpg']
index = 1 
'''
   you can easily reproduce all results shown in our case study (index from 0 to 3) 
   and the results in the appendix (index from 4 to 11).
'''

image_path = r'./image_captioning/example_images/' + image_name_list[index]
image_instance = Image.open(image_path)
display(image_instance)
```

<img src="https://github.com/yxuansu/MAGIC/blob/main/image_captioning/example_images/COCO_val2014_000000182784.jpg" width="400" height="280">


<span id='image_captioning_magic_search_result'/>

##### 5.2.5. Zero-Shot Image Captioning with Magic Search: 
Now, let's generate the image caption with magic search!
```python
'''
   setup the configurations of magic search
      k: the k in magic search
      alpha: the alpha in magic search
      beta: the beta in magic search
      decoding_len: the number of tokens to generate
'''
k, alpha, beta, decoding_len = 45, 0.1, 2.0, 16
eos_token = '<|endoftext|>'
output = generation_model.magic_search(input_ids, k, 
        alpha, decoding_len, beta, image_instance, clip, 60)
print (output)
'''
   A large cow standing in a street stall.
'''
```

<span id='image_captioning_reproduce_result'/>

##### 5.2.6. Reproduce Our Results in the Paper: 
If you would like to reproduce all the results shown in the case study and appendix of our paper, you can run this demo [file](https://github.com/yxuansu/MAGIC/blob/main/image_caption_demo.py) as

```yaml
python image_caption_demo.py
```

****

<span id='story_generation'/>

### 6. Visually Grounded Story Generation:

<span id='story_generation_experiment'/>

#### 6.1. Implementation of Experiments: 
To ensure the reproductity of our work, we provide all related resources to implement our experiments on the task of visually grounded story generation. Please refer more details [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation). 

<span id='story_generation_magic_search'/>

#### 6.2. Example Usage of Magic Search: 
In the following, we illustrate how to perform visually grounded story generation with magic search. Specifically, we show how to generate the results as shown in our case study in the paper.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19lyyMXDRNr-Op8vwUOiRmbhMxI_s3rwW?usp=sharing)

<span id='story_generation_language_model'/>

##### 6.2.1. Load Language Model: 
We first load the language model and prepare the story title as:
```python
import sys
sys.path.append(r'./story_generation/language_model')
from transformers import AutoTokenizer
from simctg import SimCTG
language_model_name = r'cambridgeltl/simctg_rocstories'
tokenizer = AutoTokenizer.from_pretrained(language_model_name)
generation_model = SimCTG(language_model_name, tokenizer.pad_token_id)
generation_model.eval()

import torch
title = 'Ice Cream Tasting <|endoftext|>'
title_tokens = tokenizer.tokenize(title)
title_id_list = tokenizer.convert_tokens_to_ids(title_tokens)
title_ids = torch.LongTensor(title_id_list).view(1,-1)
```

<span id='story_generation_CLIP'/>

##### 6.2.2. Load CLIP: 

Then, we load the CLIP model as:
```python
import sys
sys.path.append(r'./story_generation/clip')
from clip import CLIP
model_name = "openai/clip-vit-base-patch32"
clip = CLIP(model_name)
clip.eval()
```

<span id='story_generation_get_image'/>

##### 6.3.2. Get the Related Image: 
Next, let's get the images that are related to the story tile. We provide **two** ways of doing it as shown below:

<span id='story_generation_get_image_from_index'/>

###### 6.3.2.1. Retrieve from Image Index: 
The first way is to retrieve the images from a constructed image index. Before running the following commands, please make sure you have built the image index from scrath as described [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/image_index#1-build-image-index-from-scratch) or downloaded our provided image index as described [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/data#1-prepare-image-index).

After the image index is ready, we can load the image index as
```python
# build image index
import sys
sys.path.append(r'./story_generation/image_index')
from imageindex import ImageIndex
index_path = r'./story_generation/data/image_index/images_index_data/index_matrix.txt'
mapping_dict_path = r'./story_generation/data/image_index/images_index_data/mapping_dict.json'
image_folder_prefix_path = r'./story_generation/data/image_index/images/'
index = ImageIndex(index_path, mapping_dict_path, image_folder_prefix_path, clip)
```

Then, we can retrieve the top-1 images as
```python
image_name_list, image_instance_list = index.search_image(title, top_k=1)
'''
   image_name_list: the list of names of the retrieved images
   image_instance_list: the list of images that we retrieve
'''
```

Let's see the retrieved images we got
```python
from IPython.display import display
# display the top-1 image
display(image_instance_list[0])
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/avopix-284658167.jpg" width="360" height="280">


<span id='story_generation_get_image_from_example'/>

###### 6.3.2.2. Directly Load Image: 
Alternatively, if you have not prepared the image index, we have provided these the image in the repo. You can directly load it as
```python
from PIL import Image
image_name_list = ['avopix-284658167.jpg']
image_instance_list = []
for name in image_name_list:
    image_path = r'./story_generation/example_images/' + name
    image_instance = Image.open(image_path)
    image_instance_list.append(image_instance)
```

<span id='story_generation_magic_search_result'/>

##### 6.3.3. Visually Grounded Story Generation with Magic Search: 
**[Note]** Recall that, in this example, our story title is 'Ice Cream Tasting <|endoftext|>'.

Now, let's generate the story conditioned on the retrieved image
```python
from IPython.display import display
k, alpha, beta, decoding_len  = 5, 0.6, 0.15, 100
'''
   The k, alpha, beta correspond to the k, alpha, beta in magic search
'''
image_instance = image_instance_list[0]
eos_token = r'<|endoftext|>'
output, _ = generation_model.magic_search(title_ids, k, alpha, decoding_len, beta, image_instance, 
        clip, 60, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print (generated_story)
display(image_instance)
'''
   My family went to a ice cream shop. They ordered three flavors of ice cream. The first one was 
   strawberry, the second was chocolate, and the third was orange. I was excited to try all three 
   flavors. It was very good and I had a great time at the ice cream shop.
'''
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/avopix-284658167.jpg" width="360" height="280">

Then, let's see what we can get using the vanilla contrastive search **without** the image grounding.
```python
k, alpha, decoding_len  = 5, 0.6, 100
'''
   The k and alpha correspond to the k and alpha in contrastive search
'''
eos_token = r'<|endoftext|>'
output, _ = generation_model.fast_contrastive_search(title_ids, k, alpha, decoding_len, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print (generated_story)
'''
   My family went to a ice cream shop. We ordered the Ice Cream Truck. It was delicious. The customer 
   service was terrible. We had to leave for another day.
'''
```

<span id='story_generation_reproduce_result'/>

##### 6.3.4. Reproduce Our Results in the Paper: 
If you would like to reproduce all the results shown in the case study and appendix of our paper, you can run this demo [file](https://github.com/yxuansu/MAGIC/blob/main/story_generation_demo.py) as

```yaml
python story_generation_demo.py
```


****

<span id='contact'/>

### 7. Contact
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).


****

<span id='magic_elsewhere'/>

### 8. MAGIC Elsewhere
We thank the community's effort for extending MAGIC!

- [Replicate](https://replicate.com/home) has provided a great [[demo]](https://replicate.com/yxuansu/magic/examples) of MAGIC that is super easy to use. Thanks for the effort!

