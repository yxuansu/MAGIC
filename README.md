## Language Models can See: Plugging Visual Controls in Text Generation


****
## Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#news'>2. News</a>
* <a href='#citation'>3. Citation</a>
* <a href='#environment_setup'>4. Environment Setup</a>
* <a href='#image_captioning'>5. Zero-Shot Image Captioning</a>
    * <a href='#image_captioning_experiment'>5.1. Implementation of Experiments</a>
    * <a href='#image_captioning_magic_search'>5.2. Example Usage of Magic Search</a>
         * <a href='#image_captioning_language_model'>5.2.1. Load Language Model</a>
         * <a href='#image_captioning_CLIP'>5.2.2. Load CLIP</a>
         * <a href='#image_captioning_start_token'>5.2.3. Prepare Start Token</a>
         * <a href='#image_captioning_load_image'>5.2.4. Load Image</a>
         * <a href='#image_captioning_magic_search_result'>5.2.5. Zero-Shot Image Captioning with Magic Search</a>
* <a href='#story_generation'>6. Visually Grounded Story Generation</a>
    * <a href='#story_generation_experiment'>6.1. Implementation of Experiments</a>
    * <a href='#story_generation_magic_search'>6.2. Example Usage of Magic Search</a>
         * <a href='#story_generation_language_model'>6.2.1. Load Language Model</a>
         * <a href='#story_generation_get_image'>6.2.2. Get the Related Image</a>
              * <a href='#story_generation_get_image_from_index'>6.2.2.1. Retrieve from Image Index</a>
              * <a href='#story_generation_get_image_from_example'>6.2.2.2. Directly Load Image</a>
         * <a href='#story_generation_magic_search_result'>6.2.3. Visually Grounded Story Generation with Magic Search</a>
* <a href='#contact'>7. Contact</a>

****

<span id='introduction'/>

### 1. Introduction:

****

<span id='news'/>

### 2. News:

****

<span id='citation'/>

### 3. Citation:
If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```bibtex
@article{DBLP:journals/corr/abs-2202-06417,
  author    = {Yixuan Su and
               Tian Lan and
               Yan Wang and
               Dani Yogatama and
               Lingpeng Kong and
               Nigel Collier},
  title     = {A Contrastive Framework for Neural Text Generation},
  journal   = {CoRR},
  volume    = {abs/2202.06417},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.06417},
  eprinttype = {arXiv},
  eprint    = {2202.06417},
  timestamp = {Fri, 18 Feb 2022 12:23:53 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2202-06417.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
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

**[Note]** If you are looking for a python file that runs the above example, you can run this demo [file](https://github.com/yxuansu/MAGIC/blob/main/image_caption_demo.py) to see the results.

****

<span id='story_generation'/>

### 6. Visually Grounded Story Generation:

<span id='story_generation_experiment'/>

#### 6.1. Implementation of Experiments: 


<span id='story_generation_magic_search'/>

#### 6.2. Example Usage of Magic Search: 
In the following, we illustrate how to perform visually grounded story generation with magic search. Specifically, we show how to generate the results as shown in our case study in the paper.

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
title = 'The Girls <|endoftext|>'
title_tokens = tokenizer.tokenize(title)
title_id_list = tokenizer.convert_tokens_to_ids(title_tokens)
title_ids = torch.LongTensor(title_id_list).view(1,-1)
```

<span id='story_generation_get_image'/>

##### 6.2.2. Get the Related Image: 
Next, let's get the images that are related to the story tile. We provide **two** ways of doing it as shown below:

<span id='story_generation_get_image_from_index'/>

###### 6.2.2.1. Retrieve from Image Index: 
The first way is to retrieve the images from a constructed image index. Before running the following commands, please make sure you have built the image index from scrath as described [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/image_index#1-build-image-index-from-scratch) or downloaded our provided image index as described [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/data#1-prepare-image-index).

After the image index is ready, we can load the image index as
```python
# load CLIP
import sys
sys.path.append(r'./story_generation/clip')
from clip import CLIP
model_name = "openai/clip-vit-base-patch32"
clip = CLIP(model_name)
clip.eval()

# build image index
import sys
sys.path.append(r'./story_generation/image_index')
from imageindex import ImageIndex
index_path = r'./story_generation/data/image_index/images_index_data/index_matrix.txt'
mapping_dict_path = r'./story_generation/data/image_index/images_index_data/mapping_dict.json'
image_folder_prefix_path = r'./story_generation/data/image_index/images/'
index = ImageIndex(index_path, mapping_dict_path, image_folder_prefix_path, clip)
```

Then, we can retrieve the top-3 images as
```python
image_name_list, image_instance_list = index.search_image(title, top_k=3)
'''
   image_name_list: the list of names of the retrieved images
   image_instance_list: the list of images that we retrieve
'''
```

Let's see the top-3 images we got
```python
from IPython.display import display
# display the top-1 image
display(image_instance_list[0])
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/0b85a432e15c45bd55c3e83063e819c9.jpg" width="280" height="280">

```python
# display the top-2 image
display(image_instance_list[1])
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/tumblr_mg8efpjTB71rm9r1xo1_1280.jpg" width="250" height="280">

```python
# display the top-3 image
display(image_instance_list[2])
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/08a1df49b955c4522498e08ba2adc503--super-cute-dresses-simple-dresses.jpg" width="320" height="280">



<span id='story_generation_get_image_from_example'/>

###### 6.2.2.2. Directly Load Image: 
Alternatively, if you have not prepared the image index, we have provided these three images in the repo. You can directly load them as
```python
from PIL import Image
image_name_list = ['0b85a432e15c45bd55c3e83063e819c9.jpg', 'tumblr_mg8efpjTB71rm9r1xo1_1280.jpg','08a1df49b955c4522498e08ba2adc503--super-cute-dresses-simple-dresses.jpg']
image_instance_list = []
for name in image_name_list:
    image_path = r'./story_generation/example_images/' + name
    image_instance = Image.open(image_path)
    image_instance_list.append(image_instance)
```

<span id='story_generation_magic_search_result'/>

##### 6.2.3. Visually Grounded Story Generation with Magic Search: 
**[Note]** Recall that, in this example, our story title is 'The Girls <|endoftext|>'.

Now, let's generate the story conditioned on the top-1 image
```python
from IPython.display import display
k, alpha, beta, decoding_len  = 5, 0.6, 0.15, 100
image_instance = image_instance_list[0]
eos_token = r'<|endoftext|>'
output, _ = generation_model.magic_search(title_ids, k, alpha, decoding_len, beta, image_instance, 
        clip, 60, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print (generated_story)
display(image_instance)
'''
   A group of girls went to the bar. They wanted to meet up with their friends. The girls were shy 
   and didn't know each other. When they got there, they started talking. They ended up meeting up 
   at the club and had a great time.
'''
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/0b85a432e15c45bd55c3e83063e819c9.jpg" width="280" height="280">

Then, let's generate the story conditioned on the top-2 image
```python
k, alpha, beta, decoding_len  = 5, 0.6, 0.15, 100
image_instance = image_instance_list[1]
eos_token = r'<|endoftext|>'
output, _ = generation_model.magic_search(title_ids, k, alpha, decoding_len, beta, image_instance, 
        clip, 60, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print (generated_story)
display(image_instance)
'''
   The girls were excited to go to prom. They had a lot of friends and wanted to impress them. 
   They dressed as their favorite girl. The prom was over and they went home. They couldn't wait 
   to see their friends again next year in real life.
'''
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/tumblr_mg8efpjTB71rm9r1xo1_1280.jpg" width="250" height="280">

Lastly, let's generate the story conditioned on the top-3 image
```python
k, alpha, beta, decoding_len  = 5, 0.6, 0.15, 100
image_instance = image_instance_list[2]
eos_token = r'<|endoftext|>'
output, _ = generation_model.magic_search(title_ids, k, alpha, decoding_len, beta, image_instance, 
        clip, 60, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print (generated_story)
display(image_instance)
'''
   The girls were in a band. They wanted to play in front of their friends. They practiced every day.
   It was hard for them to get good at anything. They decided to disband the band in the fall.
'''
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/08a1df49b955c4522498e08ba2adc503--super-cute-dresses-simple-dresses.jpg" width="320" height="280">
****

<span id='contact'/>

### 7. Contact
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).

