## Benchmark Preparation

****
### Catalogue:
* <a href='#mscoco'>1. MSCOCO Benchmark</a>
    * <a href='#download_mscoco'>1.1. Download Post-processed Data</a>
    * <a href='#postprocess_mscoco'>1.2. Post-process Data by Yourself</a>
* <a href='#flickr30k'>2. Flickr30k Benchmark</a>
    * <a href='#download_flickr30k'>1.1. Download Post-processed Data</a>
    * <a href='#postprocess_flickr30k'>2.2. Post-process Data by Yourself</a>


****

<span id='mscoco'/>

### 1. MSCOCO Benchmark:

<span id='download_mscoco'/>

#### 1.1. Download Post-processed Data:

You can directly download our post-processed MSCOCO benchmark via this [[link]](https://drive.google.com/file/d/1J922lIqzXpLfqfWd2-F3ZI3mW59lqlBu/view?usp=sharing). After downloading, you should unzip the downloaded **mscoco.zip** file under the current directory.

> **** The resulting post-processed MSCOCO benchmark looks like:

    .
    ├── ./mscoco/                    
        ├── mscoco_train.json # Contains the training set text captions of MSCOCO
        ├── mscoco_val.json # Contains the validation set text captions of MSCOCO
        ├── mscoco_test.json # Contains the test set text captions of MSCOCO
        └── test_images # Contains the test set images of MSCOCO
        
The json files contain a list of dictory, where the data format of each dictionary is:

```yaml
{  
   "split": Indicating which split (train, val, or test) the data instance belongs to.
   "image_name": The name of the corresponding image.
   "file_path": Where to find the corresponding image from the raw MSCOCO files.
   "captions": A list of captions that the data instance contains.
}
```

<span id='postprocess_mscoco'/>


#### 1.2. Post-process Data by Yourself:

We also provide the scripts that help you recreate the post-processed MSCOCO benchmark.

- (1) Download the data split following instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/data/raw_data).
- (2) Download the raw images following instructions [[here]](https://github.com/yxuansu/MAGIC/blob/main/image_captioning/data/raw_images/README.md#1-download-mscoco-raw-images).

Afterwards, run the following command:
```yaml
python process_mscoco.py
```


****

<span id='flickr30k'/>

### 2. Flickr30k Benchmark:

<span id='download_flickr30k'/>

#### 1.1. Download Post-processed Data:

You can directly download our post-processed Flickr30k benchmark via this [[link]](https://drive.google.com/file/d/1i8-v-U3qlhK9uW_RzV3iV8IRJlKTpcBZ/view?usp=sharing). After downloading, you should unzip the downloaded **flickr30k.zip** file under the current directory.

> **** The resulting post-processed Flickr30k benchmark looks like:

    .
    ├── ./flickr30k/                    
        ├── flickr30k_train.json # Contains the training set text captions of Flickr30k
        ├── flickr30k_val.json # Contains the validation set text captions of Flickr30k
        ├── flickr30k_test.json # Contains the test set text captions of Flickr30k
        └── test_images # Contains the test set images of Flickr30k
        
The json files contain a list of dictory, where the data format of each dictionary is:

```yaml
{  
   "split": Indicating which split (train, val, or test) the data instance belongs to.
   "image_name": The name of the corresponding image.
   "captions": A list of captions that the data instance contains.
}
```


<span id='postprocess_flickr30k'/>


#### 2.2. Post-process Data by Yourself:

We also provide the scripts that help you recreate the post-processed Flickr30k benchmark.

- (1) Download the data split following instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/data/raw_data).
- (2) Download the raw images following instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/data/raw_images#2-download-flickr30k-raw-images).

Afterwards, run the following command:
```yaml
python process_flickr30k.py
```









