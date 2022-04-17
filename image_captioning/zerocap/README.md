# Pytorch Implementation of [Zero-Shot Image-to-Text Generation for Visual-Semantic Arithmetic](https://arxiv.org/abs/2111.14447) [CVPR 2022]
[[Paper]](https://arxiv.org/abs/2111.14447) [[Notebook]](https://www.kaggle.com/yoavstau/zero-shot-image-to-text/notebook) [[Caption Demo]](https://replicate.com/yoadtew/zero-shot-image-to-text) [[Arithmetic Demo]](https://replicate.com/yoadtew/arithmetic)

## Approach
![](git_images/Architecture.jpg)

## Example of capabilities
![](git_images/teaser.jpg)

## Example of Visual-Semantic Arithmetic
![](git_images/relations.jpg)

## Usage

### Prepare the requirments

```bash
pip install -r requirements.txt
```

### To run captioning on MSCOCO benchmark

```bash
./mscoco_zerocap.sh
```

### To run captioning on Flickr30k benchmark

```bash
./flickr30k_zerocap.sh
```

### To run captioning on cross-domain setting (Flickr30k LM to MSCOCO images)

```bash
./flickr30k_to_mscoco_zerocap.sh
```

### To run captioning on cross-domain setting (MSCOCO LM to Flickr30k images)

```bash
./mscoco_to_flickr30k_zerocap.sh
```

## Citation
Please cite our work if you use it in your research:
```
@article{tewel2021zero,
  title={Zero-Shot Image-to-Text Generation for Visual-Semantic Arithmetic},
  author={Tewel, Yoad and Shalev, Yoav and Schwartz, Idan and Wolf, Lior},
  journal={arXiv preprint arXiv:2111.14447},
  year={2021}
}
```
