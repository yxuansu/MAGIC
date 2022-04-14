## Unsupervised Domain Adaptation of Language Model
****
### Catalogue:
* <a href='#mscoco'>1. MSCOCO Benchmark</a>
    * <a href='#mscoco_data_preparation'>1.1. MSCOCO Data Preparation</a>
    * <a href='#mscoco_training'>1.2. Unsupervised Domain Adaptation on MSCOCO</a>
* <a href='#flickr30k'>2. Flickr30k Benchmark</a>
    * <a href='#flickr30k_data_preparation'>2.1. Flickr30k Data Preparation</a>
    * <a href='#flickr30k_training'>2.2. Unsupervised Domain Adaptation on Flickr30k</a>
* <a href='#unsupervised_baselines'>3. Unsupervised Baselines</a>
    * <a href='#contrastive_search'>3.1. Contrastive Search</a>
    * <a href='#top_k_sampling'>3.2. Top-k Sampling</a>
    * <a href='#nucleus_sampling'>3.3. Nucleus Sampling</a> 

****
<span id='mscoco'/>

#### 1. MSCOCO Benchmark:

We first describe how to perform unsupervised domain adaptation of language model on the text corpus of MSCOCO benchmark.

<span id='mscoco_data_preparation'/>

##### 1.1. MSCOCO Data Preparation:

To prepare the MSCOCO benchmark, please follow the instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/data#1-mscoco-benchmark).

<span id='mscoco_training'/>

##### 1.2.Unsupervised Domain Adaptation on MSCOCO:
After preparing the MSCOCO data, run the following command to train the language model.
```yaml
chmod +x ./train_mscoco.sh
./train_mscoco.sh
```
The arguments are as follows:
* `--model_name`: The name of huggingface pre-trained gpt model (e.g. gpt2, gpt-large).
* `--train_path`: The file path of training set.
* `--dev_path`: The file path of validation set.
* `--test_path`: The file path of test set.
* `--add_eos_token_to_data`: Whether adding an eos token at the end of text sequence.
* `--margin`: The contrastive margin $\rho$.
* `--max_len`: The maximum length of training samples.
* `--number_of_gpu`: The number of available GPUs.
* `--batch_size_per_gpu`: The batch size for each GPU.
* `--gradient_accumulation_steps`: How many forward computations between two gradient updates.
* `--effective_batch_size`: The overall batch size. It equals to batch_size_per_gpu x gradient_accumulation_steps x number_of_gpu.
* `--total_steps`: The number of total gradient update steps.
* `--print_every`: Have many steps to show the intermediate results.
* `--save_every`: How many steps to save one checkpoint.
* `--learning_rate`: The learning rate.
* `--save_path_prefix`: Where to save the checkpoints.

****
<span id='flickr30k'/>

#### 2. Flickr30k Benchmark:

We then describe how to perform unsupervised domain adaptation of language model on the text corpus of Flickr30k benchmark.

<span id='flickr30k_data_preparation'/>

##### 2.1. Flickr30k Data Preparation:

To prepare the Flickr30k benchmark, please follow the instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/data#2-flickr30k-benchmark).

<span id='flickr30k_training'/>

##### 2.2. Unsupervised Domain Adaptation on Flickr30k:
After preparing the Flickr30k data, run the following command to train the language model.
```yaml
chmod +x ./train_flickr30k.sh
./train_flickr30k.sh
```

****
<span id='unsupervised_baselines'/>

#### 3. Unsupervised Baselines:

Here, we illustrate how to use the language model to perform unsupervised baselines as described in our paper. Note that, all these methods are **unsupervised** as the language model is a text-only model and does not take image as input. 

```python
# first, load the language model
import torch
from simctg import SimCTG
sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
# we use the language model adapted on MSCOCO as an example.
language_model_name = r'cambridgeltl/magic_mscoco'
generation_model = SimCTG(language_model_name, sos_token, pad_token)
generation_model.eval()

# then, prepare the input ids. Note that, the text is always generated from the same start of sentence token.
tokens = generation_model.tokenizer.tokenize(sos_token)
input_ids = generation_model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)
```

<span id='contrastive_search'/>

##### 3.1. Contrastive Search :
```python
'''
   use contrastive search to generate the result.
   note that, contrastive search is a deterministic decoding method, thus the generated text is always the same.
'''
beam_width, alpha, decoding_len = 45, 0.1, 16
output_text = generation_model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len)
print (output_text)
'''
   A man is riding a skateboard down a street.
'''
```
The arguments are as follows:
* `--input_ids`: The id of the start of sentence token.
* `--beam_width`: k in the contrastive search.
* `--alpha`: alpha in the contrastive search.
* `--decoding_len`: Number of tokens to generate.

<span id='top_k_sampling'/>

##### 3.2. Top-k Sampling :
```python
'''
   use top-k sampling to generate the result.
   note that, the this method is a stochastic method, thus the generated text is always different.
'''
top_k, decoding_len = 40, 16
output_text = generation_model.top_k_sampling(input_ids, top_k, decoding_len)
print (output_text)
'''
   some very different types of vases with flowers together
'''
```
The arguments are as follows:
* `--input_ids`: The id of the start of sentence token.
* `--k`: The k in top-k sampling.
* `--decoding_len`: Number of tokens to generate.

<span id='nucleus_sampling'/>

##### 3.3. Nucleus Sampling :
```python
'''
   use nucleus sampling to generate the result.
   note that, the this method is a stochastic method, thus the generated text is always different.
'''
nucleus_p, decoding_len = 0.95, 16
output_text = generation_model.nucleus_sampling(input_ids, nucleus_p, decoding_len)
print (output_text)
'''
   Two young girls enjoying a hot dog hot dog bun.
'''
```
The arguments are as follows:
* `--input_ids`: The id of the start of sentence token.
* `--nucleus_p`: The probability in nucleus sampling.
* `--decoding_len`: Number of tokens to generate.









