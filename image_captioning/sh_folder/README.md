## Details of How to Implement the Inference Code

In this folder, we illustrate how to conduct inference with different methods.

****
### Catalogue:
* <a href='#magic_search'>1. Magic Search</a>
    * <a href='#in_domain_magic_search'>1.1. In Domain Experiment</a>
    * <a href='#cross_domain_magic_search'>1.2. Cross Domain Experiment</a>


****

<span id='magic_search'/>

### 1. Magic Search:

<span id='in_domain_magic_search'/>

#### 1.1. In Domain Experiment:

To perform in domain experiment with magic search, please run the following command:
```yaml
chmod +x ./X_magic.sh
./X_magic.sh
```

Here, X is in ['mscoco', 'flickr30k'] and the arguments are as follows:
* `--input_ids`: The id of the start of sentence token.
* `--beam_width`: k in the contrastive search.
* `--alpha`: alpha in the contrastive search.
* `--decoding_len`: Number of tokens to generate.

<span id='cross_domain_magic_search'/>

#### 1.2. Cross Domain Experiment:
