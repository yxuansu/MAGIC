### Post-processing Benchmarks

#### 1. MSCOCO Benchmark
To post-processing the MSCOCO benchmark, please first download the data split following instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/data/raw_data). Then, downloading the raw images following instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/data/raw_images).

After downloading the raw data, run the following command:
```yaml
python process_mscoco.py
```

> **** The resulting post-processed MSCOCO benchmark looks like:

    .
    ├── ./mscoco/                    
        ├── mscoco_train.json # Contains the training set text captions of MSCOCO
        ├── mscoco_val.json # Contains the validation set text captions of MSCOCO
        ├── mscoco_test.json # Contains the test set text captions of MSCOCO
        └── test_images # Contains the test set images of MSCOCO
