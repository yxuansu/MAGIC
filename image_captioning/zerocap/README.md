### Our Implementation of the ZeroCap Baseline Model 

****
### Catalogue:
* <a href='#environment'>1. Environment Preparation</a>
* <a href='#mscoco'>2. Image Captioning on MSCOCO</a>
* <a href='#flickr30k'>3. Image Captioning on Flickr30k</a>
* <a href='#flickr30k_to_mscoco'>4. Cross Domain Image Captioning on MSCOCO</a>
* <a href='#mscoco_to_flickr30k'>5. Cross Domain Image Captioning on Flickr30k</a>
* <a href='#citation'>6. Citation</a>
* <a href='#acknowledgements'>7. Acknowledgements</a>

****

<span id='environment'/>

#### 1. Environment Preparation:
To install the correct environment, please run the following command:
```yaml
pip install -r requirements.txt
```

****

<span id='mscoco'/>

#### 2. Image Captioning on MSCOCO:
To perform image captioning on MSCOCO, please run the following command:
```yaml
chmod +x ./mscoco_zerocap.sh
./mscoco_zerocap.sh
```

****

<span id='flickr30k'/>

#### 3. Image Captioning on Flickr30k:
To perform image captioning on Flickr30k, please run the following command:
```yaml
chmod +x ./flickr30k_zerocap.sh
./flickr30k_zerocap.sh
```

****

<span id='flickr30k_to_mscoco'/>

#### 4. Cross Domain Image Captioning on MSCOCO:
To perform image captioning on MSCOCO with the language model from Flickr30k domain, please run the following command:
```yaml
chmod +x ./flickr30k_to_mscoco_zerocap.sh
./flickr30k_to_mscoco_zerocap.sh
```

****

<span id='mscoco_to_flickr30k'/>

#### 5. Cross Domain Image Captioning on Flickr30k:
To perform image captioning on Flickr30k with the language model from MSCOCO domain, please run the following command:
```yaml
chmod +x ./mscoco_to_flickr30k_zerocap.sh
./mscoco_to_flickr30k_zerocap.sh
```

****

<span id='citation'/>

#### 6. Citation:
If you find our code helpful, please cite the original paper as

```bibtex
@article{tewel2021zero,
  title={Zero-Shot Image-to-Text Generation for Visual-Semantic Arithmetic},
  author={Tewel, Yoad and Shalev, Yoav and Schwartz, Idan and Wolf, Lior},
  journal={arXiv preprint arXiv:2111.14447},
  year={2021}
}
```

****

<span id='acknowledgements'/>

#### 7. Acknowledgements:
We thank the authors for releasing their code. Our reimplementation of the baseline is based on their original codebase [[here]](https://github.com/yoadtew/zero-shot-image-to-text).

