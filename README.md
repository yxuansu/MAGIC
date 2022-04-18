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
         * <a href='#image_captioning_magic_search'>5.2.5. Zero-Shot Image Captioning with Magic Search</a>
* <a href='#story_generation'>6. Visually Grounded Story Generation</a>
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
Note that, the language model always starts generation with a special start of sentence token. Here, we prepare the input ids of the start token.
```python
import torch
sos_token = r'<-start_of_text->'
start_token = generation_model.tokenizer.tokenize(sos_token)
start_token_id = generation_model.tokenizer.convert_tokens_to_ids(start_token)
input_ids = torch.LongTensor(start_token_id).view(1,-1)
```

<span id='image_captioning_load_image'/>

##### 5.2.4. Load Image: 

```python
```

<span id='image_captioning_magic_search'/>

##### 5.2.5. Zero-Shot Image Captioning with Magic Search: 

```python
```


****

<span id='contact'/>

### 7. Contact
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).

