## Unsupervised Domain Adaptation of Language Model
****
### Catalogue:
* <a href='#mscoco'>1. MSCOCO Benchmark</a>
    * <a href='#mscoco_data_preparation'>1.1. MSCOCO Data Preparation</a>
    * <a href='#mscoco_training'>1.2. Unsupervised Domain Adaptation on MSCOCO</a>
* <a href='#flickr30k'>2. Flickr30k Benchmark</a>
    * <a href='#flickr30k_data_preparation'>1.1. Flickr30k Data Preparation</a>
    * <a href='#flickr30k_training'>1.2. Unsupervised Domain Adaptation on Flickr30k</a>
    
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





