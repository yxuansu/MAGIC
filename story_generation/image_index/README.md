## Image Index
Here, we illustrate how to construct the image index from scratch. In addition, we provide an example usage session to demonstrate how to use the constructed image index to retrieve images based on the given text.

****
### Catalogue:
* <a href='#build_index'>1. Build Image Index from Scratch</a>
* <a href='#example_usage'>2. Example Usage</a>
    * <a href='#load_clip'>2.1. Load CLIP</a>
    * <a href='#load_index'>2.2. Load Image Index</a>
    * <a href='#example_1'>2.3. Example 1</a>
    * <a href='#example_2'>2.4. Example 2</a>

****

<span id='build_index'/>

### 1. Build Image Index from Scratch:
Given a set of raw images, we use CLIP to compute their representations and store them in several files for an easy access. To build your own image index, you should run the following command:
```yaml
chmod +x ./build_index.sh
./build_index.sh
```
The arguments are as follows:
* `--clip_name`: The configuration of huggingface pre-trained CLIP model (e.g. (i) openai/clip-vit-base-patch32, (ii) openai/clip-vit-base-patch14, and (iii) openai/clip-vit-large-patch14).
* `--image_file_prefix_path`: The directory that you store your raw images.
* `--save_index_prefix`: The directory that stores your constructed index files.
* `--save_index_name`: The name used to save the representation matrix.
* `--save_image_name_dict`: The name used to save the mapping dictionary.
* `--batch_size`: The inference batch size.

**[Note]** Before constructing the image index, please make sure you have prepared the raw images as demonstrated [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/data#11-download-raw-images).

****

<span id='example_usage'/>

### 2. Example Usage:
After constructing the image index, we can retrieve the related images based on the text input. In the following, we provide two examples on how to perform the image retrieval process.

<span id='load_clip'/>

#### 2.1. Load CLIP:
We first load the off-the-shelf CLIP model as:
```python
import sys
sys.path.append(r'../clip')
from clip import CLIP
model_name = "openai/clip-vit-base-patch32"
clip = CLIP(model_name)
clip.eval()
```

<span id='load_index'/>

#### 2.2. Load Image Index:
Then, we load the constructed image index as:
```python
from imageindex import ImageIndex
index_path = r'../data/image_index/images_index_data/index_matrix.txt'
mapping_dict_path = r'../data/image_index/images_index_data/mapping_dict.json'
image_folder_prefix_path = r'../data/image_index/images/'
index = ImageIndex(index_path, mapping_dict_path, image_folder_prefix_path, clip)
```

**[Note]** The arguments index_path, mapping_dict_path, and image_folder_prefix_path should be the same as you set in the build_index.sh script. 

<span id='example_1'/>

#### 2.3. Example 1:
Here, we provide the first example.
```python
text = 'Maid Of Honor'
image_name_list, image_instance_list = index.search_image(text, top_k=3)
'''
   image_name_list: The list of names of the retrieved images.
   image_instance_list: The list of instances of the retrieved images.
'''

from IPython.display import display # to display images
for image_instance in image_instance_list:
    display(image_instance)
```
The search_image() method takes two arguments:
* `--text`: The input text.
* `--top_k`: The number of retrieved images.

The retrieved images are:

<img src="https://github.com/yxuansu/MAGIC/tree/main/story_generation/image_index/example_images/3a8faacd322e262dbe1de2e837508449--daddys-little-girls-baby-girls.jpg" width="400" height="280">






    * <a href='#example_2'>2.4. Example 2</a>


