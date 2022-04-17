## Inference Results
This directory contains the results of all evaluated methods for the image captioning task.

> ****  The structure of the directory looks like:

    .
    ├── ./mscoco/  # results on MSCOCO benchmark               
        ├── magic_result.json # The results of our magic approach
        ├── baselines # The directory that contains all baseline results
            ├── mscoco_test.json # Contains the test set text captions of MSCOCO
            └── test_images # Contains the test set images of MSCOCO
    ├── ./flickr30k/  # results on Flickr30k benchmark
    ├── ./flickr30k_model_to_mscoco/ # results of cross domain image captioning on MSCOCO benchmark
    ├── ./mscoco_model_to_flickr30k/ # results of cross domain image captioning on Flickr30k benchmark
