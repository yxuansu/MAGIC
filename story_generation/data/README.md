## Data Preparation
Here, we provide the test set of ROCStories benchmark that we use in our experiments. In the file, each line follows the format below:
```yaml
story tile + \t + human-written story
```

In the following, we illustrate how to build the image index from which we retrieve images based on the given story title.

****
### Catalogue:
* <a href='#prepare_image_index'>1. Prepare Image Index</a>
    * <a href='#download_images'>1.1 Download Raw Images</a>
    * <a href='#download_index'>1.2 Download Index Files</a>
* <a href='#build_own_index'>2. Build Your Own Image Index</a>

****

<span id='prepare_image_index'/>

#### 1. Prepare Image Index:

<span id='download_images'/>

##### 1.1 Download Raw Images:
Please download the images via [[google drive]](https://drive.google.com/file/d/1yQc9hFOQGfqsa1vG6m0SRCZAdaDwy9Qi/view?usp=sharing). The file has a size around 30GB. After downloading, run the following command to unzip it **under the ./image_index/ directory**.
```yaml
unzip images.zip
```

After unzipping, the directory ./image_index/images/ should contain around **330k** images which is a subset of the original [[conceptualcaptions dataset]](https://www.conceptualcaptions.com/home). 

> **** Some examples are listed as below:

    .
    ├── ./image_index/images/                       
        ├── __57.jpg         
        ├── _1_bird_jpg.jpg.size.custom.crop.1086x727.jpg 
        ├── _02_big.jpg 
        ├── _2_eastsuite_1_jpg.jpg.size.custom.crop.850x567.jpg 
        ├── _05-pebble-beach-concept-lawn-2017-1.jpg 
        └── ...


<span id='download_index'/>

##### 1.2 Download Index Files:
After downloading raw images, please download the index files via [[google drive]](https://drive.google.com/file/d/13qCKHdGuV1Rp3KbWHRSS6FStd-fnF48i/view). Please unzip the file **under the ./image_index/ directory**.
```yaml
unzip images_index_data.zip
```

> **** After unzipping, the directory ../image_index/images_index_data/ should contain two files:

    .
    ├── ./image_index/images_index_data/                      
        ├── index_matrix.txt         
        └── mapping_dict.json

**File Details:**

(i) index_matrix.txt: This file contains CLIP representations of the raw images in the directory of ../image_index/images/. Each line contains 512 float numbers corresponding to one 512-dimensional vector representation of an image.

(ii) mapping_dict.json: This file contains a dictionary in which key is row number and value is image name. Some examples are:
```yaml
{  
    "0": "5f312abec28ffbae887376f30aa41cef.jpg",
    "1": "6c1a612337c05b4c13cbcf66739cf796--the-coronation-sissi.jpg",
    "2": "peanut-butter-cup-protein-overnight-oats-1131.jpg",
    "3": "rings-on-the-blue-water-j23f17.jpg",
    "4": "old-rotten-fishing-boat-lying-at-the-beach-trees-are-already-growing-drf2a3.jpg",
    "5": "st-bernard-male-dog-running-across-a-stubble-field-dgy24w.jpg",
    ...
}
```

The first entry in the above example means that the "0"-th line of index_matrix.txt corresponds to the representation of the image "./image_index/images/5f312abec28ffbae887376f30aa41cef.jpg" produced by CLIP.

**[Note]** The number of key-value pairs in mapping_dict.json equals to the number of lines in index_matrix.txt. Therefore, we can find the unique representation of every raw image in ./images with the help of these two files.

****

<span id='build_own_index'/>

#### 2. Build Your Own Image Index:
If you would like to rebuild a new image index with CLIP, you should first prepare a set of your own images and put them under a directory as described in section 1.1. Then, you can build your own index following our provided instructions [[here]](https://github.com/yxuansu/MAGIC/blob/main/story_generation/image_index/README.md#1-build-image-index-from-scratch).

