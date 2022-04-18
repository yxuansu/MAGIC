## Image Index
Here, we illustrate how to construct the image index from scratch. In addition, we provide an example usage session to demonstrate how to use the constructed image index to retrieve images based on the given text.

****
### Catalogue:
* <a href='#build_index'>1. Build Image Index from Scratch</a>
* <a href='#example_usage'>2. Example Usage</a>
    * <a href='#load_clip'>2.1. Load CLIP</a>
    * <a href='#postprocess_flickr30k'>2.2. Post-process Data by Yourself</a>

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

**[Note]** Before constructing the image index, please make sure you have prepared the raw images as demonstrated [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/data#11-download-raw-images)

****

<span id='example_usage'/>

### 2. Example Usage:
After constructing the image index, we can retrieve the 


