## Benchmark Preparation

****
### Catalogue:
* <a href='#mscoco'>1. MSCOCO</a>
    * <a href='#download_mscoco'>1.1. Download Post-processed Data</a>


****

<span id='mscoco'/>

#### 1. MSCOCO:

<span id='download_mscoco'/>

##### 1.1. Download Post-processed Data:

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




To post-processing the MSCOCO benchmark, please first download the data split following instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/data/raw_data). Then, downloading the raw images following instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/data/raw_images).

After downloading the raw data, run the following command:
```yaml
python process_mscoco.py
```







