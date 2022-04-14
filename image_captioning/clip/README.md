## CLIP
This folder illustrates how to use CLIP to build text index and to conduct cross-modal retrieval baseline.

****
## Catalogue:
* <a href='#index'>1. Build Text Index</a>
    * <a href='#mscoco'>1.1. Build Text Index for MSCOCO</a>
        * <a href='#download_mscoco_index'>1.1.1. Download Our Built Index</a>
        * <a href='#process_mscoco_index'>1.1.2. Construct the Index by Yourself</a>

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


