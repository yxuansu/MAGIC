## CLIP
This folder illustrates how to use CLIP to build text index and to conduct cross-modal retrieval baseline.

****
## Catalogue:
* <a href='#index'>1. Build Text Index</a>
    * <a href='#mscoco'>1.1. Build Text Index for MSCOCO</a>
        * <a href='#download_mscoco_index'>1.1.1. Download Our Built Index</a>
        * <a href='#process_mscoco_index'>1.1.2. Construct the Index by Yourself</a>
    * <a href='#flickr30k'>1.2. Build Text Index for Flickr30k</a>
        * <a href='#download_flickr30k_index'>1.2.1. Download Our Built Index</a>
        * <a href='#process_flickr30k_index'>1.2.2. Construct the Index by Yourself</a>
* <a href='#baseline'>2. CLIP Retrieval Baseline</a>
    * <a href='#in_domain_baseline'>2.1. In Domain CLIP Retrieval</a>
    * <a href='#cross_domain_baseline'>2.2. Cross Domain CLIP Retrieval</a>

****

<span id='index'/>

### 1. Build Text Index:
We show how to build the text index, from which the caption is retrieved from, using CLIP.

<span id='mscoco'/>

#### 1.1. Build Text Index for MSCOCO:
First, we demonstrate how to build text index for MSCOCO.

<span id='download_mscoco_index'/>

#### 1.1.1. Download Our Post-processed Index:
We share our built index for MSCOCO via this [[link]](https://drive.google.com/file/d/1Dx_RPeAmydS6ZYuiJ-dLlK9-DjDZkxAh/view?usp=sharing). After downloading, unzip the downloaded file **mscoco_index.zip** under the current directory.

> **** The resulting directory looks like:

    .
    ├── ./mscoco_index/                    
        ├── index_matrix.txt # The file that stores the representations of captions from the training set of MSCOCO. Each row is a vector that corresponds to a specific caption from the training set.
        └── text_mapping.json # The file that stores the mappings between the representation and the corresponding caption.

<span id='process_mscoco_index'/>

#### 1.1.2. Construct the Index by Yourself:

You can also rebuild the index by yourself. First, you should make sure you have downloaded the MSCOCO data following instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/data#1-mscoco-benchmark). Then, you can run the following command to build the index.
```yaml
chmod +x ./build_mscoco_index.sh
./build_mscoco_index.sh
```
The arguments are as follows:
* `--clip_name`: The configuration of the pre-trained CLIP model from huggingface.
* `--text_file_path`: Where the training text corpus stores.
* `--save_index_prefix`: In which directory you would like to store your index files.
* `--save_index_name`: The saved name of the caption representations.
* `--save_mapping_dict_name`: The saved name of the mapping dictionary between representations and captions.
* `--batch_size`: The inference batch size.


<span id='flickr30k'/>

#### 1.2. Build Text Index for Flickr30k:
Next, we demonstrate how to build text index for Flickr30k.

<span id='download_flickr30k_index'/>

#### 1.2.1. Download Our Post-processed Index:
We share our built index for Flickr30k via this [[link]](https://drive.google.com/file/d/1hS58_ir5pdZZPckApCtlz2RyasCQbrPf/view?usp=sharing). After downloading, unzip the downloaded file **flickr30k_index.zip** under the current directory.

> **** The resulting directory looks like:

    .
    ├── ./flickr30k_index/                    
        ├── index_matrix.txt # The file that stores the representations of captions from the training set of Flickr30k. Each row is a vector that corresponds to a specific caption from the training set.
        └── text_mapping.json # The file that stores the mappings between the representation and the corresponding caption.

<span id='process_flickr30k_index'/>

#### 1.2.2. Construct the Index by Yourself:

You can also rebuild the index by yourself. First, you should make sure you have downloaded the Flickr30k data following instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/data#2-flickr30k-benchmark). Then, you can run the following command to build the index.
```yaml
chmod +x ./build_flickr30k_index.sh
./build_flickr30k_index.sh
```
The arguments are as follows:
* `--clip_name`: The configuration of the pre-trained CLIP model from huggingface.
* `--text_file_path`: Where the training text corpus stores.
* `--save_index_prefix`: In which directory you would like to store your index files.
* `--save_index_name`: The saved name of the caption representations.
* `--save_mapping_dict_name`: The saved name of the mapping dictionary between representations and captions.
* `--batch_size`: The inference batch size.

****

<span id='baseline'/>

### 2. CLIP Retrieval Baseline:
Here, we show how to conduct the CLIP retrieval baseline.

<span id='in_domain_baseline'/>

#### 2.1. In Domain CLIP Retrieval:
To retrieve the captions from the in domain training set, you should run the following command:
```yaml
chmod +x ./X_clip_retrieval.sh
./X_clip_retrieval.sh
```
Here, X is in ['mscoco', 'flickr30k'] which corresponds for the MSCOCO and Flickr30k benchmarks.

The arguments are as follows:
* `--clip_name`: The configuration of the pre-trained CLIP model from huggingface.
* `--test_image_prefix_path`: Where the test set images stores.
* `--test_path`: Where the reference test captions file stores.
* `--index_matrix_path`: The path of the representation index file.
* `--mapping_dict_path`: The path of the mapping dictionary between representations and captions.
* `--save_path_prefix`: Where to save the inferenced result.
* `--save_name`: The saved name of the inferenced result.

**[Note]** As we are conducting in domain CLIP retrieval, the test images and the caption index should come from the same benchmark.


<span id='cross_domain_baseline'/>

#### 2.2. Cross Domain CLIP Retrieval:
To retrieve the captions from the cross domain training set, you should run the following command:
```yaml
chmod +x ./source_X_target_Y_clip_retrieval.sh
./source_X_target_Y_clip_retrieval.sh
```
Here, X is the source domain from ['mscoco', 'flickr30k'] and Y is the target domain from ['flickr30k', 'mscoco'].

The arguments are as follows:
* `--clip_name`: The configuration of the pre-trained CLIP model from huggingface.
* `--test_image_prefix_path`: Where the test set images stores.
* `--test_path`: Where the reference test captions file stores.
* `--index_matrix_path`: The path of the representation index file.
* `--mapping_dict_path`: The path of the mapping dictionary between representations and captions.
* `--save_path_prefix`: Where to save the inferenced result.
* `--save_name`: The saved name of the inferenced result.

**[Note]** As we are conducting cross domain CLIP retrieval, the test images and the caption index should come from **different** benchmarks.
