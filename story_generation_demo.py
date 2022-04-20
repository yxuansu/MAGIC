import sys
sys.path.append(r'./story_generation/language_model')
from transformers import AutoTokenizer
from simctg import SimCTG
language_model_name = r'cambridgeltl/simctg_rocstories'
tokenizer = AutoTokenizer.from_pretrained(language_model_name)
generation_model = SimCTG(language_model_name, tokenizer.pad_token_id)
generation_model.eval()

import torch
title = 'The Girls <|endoftext|>'
title_tokens = tokenizer.tokenize(title)
title_id_list = tokenizer.convert_tokens_to_ids(title_tokens)
title_ids = torch.LongTensor(title_id_list).view(1,-1)

import sys
sys.path.append(r'./story_generation/clip')
from clip import CLIP
model_name = "openai/clip-vit-base-patch32"
clip = CLIP(model_name)
clip.eval()

from PIL import Image
image_name_list = ['0b85a432e15c45bd55c3e83063e819c9.jpg', 'tumblr_mg8efpjTB71rm9r1xo1_1280.jpg','08a1df49b955c4522498e08ba2adc503--super-cute-dresses-simple-dresses.jpg']
image_instance_list = []
image_path_list = []
for name in image_name_list:
    image_path = r'./story_generation/example_images/' + name
    image_path_list.append(image_path)
    image_instance = Image.open(image_path)
    image_instance_list.append(image_instance)

k, alpha, beta, decoding_len  = 5, 0.6, 0.15, 100
'''
   The k, alpha, beta correspond to the k, alpha, beta in magic search
'''
image_instance = image_instance_list[0]
eos_token = r'<|endoftext|>'
output, _ = generation_model.magic_search(title_ids, k, alpha, decoding_len, beta, image_instance, 
        clip, 60, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print ('------------------- The story conditioned on the top-1 image is ----------------------------')
print (generated_story)

print ('------------------- The story conditioned on the top-2 image is ----------------------------')
k, alpha, beta, decoding_len  = 5, 0.6, 0.15, 100
'''
   The k, alpha, beta correspond to the k, alpha, beta in magic search
'''
image_instance = image_instance_list[1]
eos_token = r'<|endoftext|>'
output, _ = generation_model.magic_search(title_ids, k, alpha, decoding_len, beta, image_instance, 
        clip, 60, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print (generated_story)

print ('------------------- The story conditioned on the top-3 image is ----------------------------')
k, alpha, beta, decoding_len  = 5, 0.6, 0.15, 100
'''
   The k, alpha, beta correspond to the k, alpha, beta in magic search
'''
image_instance = image_instance_list[2]
eos_token = r'<|endoftext|>'
output, _ = generation_model.magic_search(title_ids, k, alpha, decoding_len, beta, image_instance, 
        clip, 60, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print (generated_story)


'''
------------------- The story conditioned on the top-1 image is ----------------------------
A group of girls went to the bar. They wanted to meet up with their friends. The girls were shy and didn't know each other. When they got there, they started talking. They ended up meeting up at the club and had a great time.
------------------- The story conditioned on the top-2 image is ----------------------------
The girls were excited to go to prom. They had a lot of friends and wanted to impress them. They dressed as their favorite girl. The prom was over and they went home. They couldn't wait to see their friends again next year in real life.
------------------- The story conditioned on the top-3 image is ----------------------------
The girls were in a band. They wanted to play in front of their friends. They practiced every day. It was hard for them to get good at anything. They decided to disband the band in the fall.
'''
