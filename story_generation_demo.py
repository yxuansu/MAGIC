import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

print ('Loading language model...')
import sys
sys.path.append(r'./story_generation/language_model')
from transformers import AutoTokenizer
from simctg import SimCTG
language_model_name = r'cambridgeltl/simctg_rocstories'
tokenizer = AutoTokenizer.from_pretrained(language_model_name)
generation_model = SimCTG(language_model_name, tokenizer.pad_token_id)
generation_model = generation_model.to(device)
generation_model.eval()
print ('Language model loaded.')

from PIL import Image
print ('Loading CLIP...')
sys.path.append(r'./story_generation/clip')
from clip import CLIP
model_name = "openai/clip-vit-base-patch32"
clip = CLIP(model_name)
clip = clip.to(device)
clip.eval()
print ('CLIP loaded.')

'''
    The first two instances correspond to our case study in the main content of the paper.
    The rest of the instances correspond to our case study in the appendix.
'''
title_list = ['Ice Cream Tasting <|endoftext|>', 'Sand Volleyball <|endoftext|>', 'Rainstorm <|endoftext|>', 
            'French Braid <|endoftext|>', 'The Hair Clump <|endoftext|>', 'Pig <|endoftext|>']


image_name_list = ['avopix-284658167.jpg', 'stock-photo-group-of-friends-women-and-men-playing-beach-volleyball-one-in-front-doing-tricks-to-the-ball-62655943.jpg',
                'stock-vector-water-drops-on-the-window-glass-206158339.jpg','43bdb53f0b81082701c0ddefe8e46395--loose-side-braids-big-braids.jpg',
                'c46436b3b752aab8d348f83c91fbeafe--ashton-irwin-hair-crushes.jpg','stock-photo-pig-who-is-represented-on-a-white-background-71087542.jpg']

k, alpha, beta, decoding_len  = 5, 0.6, 0.15, 100
eos_token = r'<|endoftext|>'

for idx in range(len(title_list)):
    title = title_list[idx]
    title_tokens = tokenizer.tokenize(title)
    title_id_list = tokenizer.convert_tokens_to_ids(title_tokens)
    title_ids = torch.LongTensor(title_id_list).view(1,-1)

    output, _ = generation_model.fast_contrastive_search(title_ids, k, alpha, decoding_len, eos_token)
    _, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
    
    print ('------ Story Title is ------')
    print (title.strip(eos_token).strip())
    print ('------ Contrastive Search Result is ------')
    print (generated_story)

    image_path = r'./story_generation/example_images/' + '/' + image_name_list[idx]
    image_instance = Image.open(image_path)
    output, _ = generation_model.magic_search(title_ids, k, alpha, decoding_len, beta, image_instance, 
        clip, 60, eos_token)
    _, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)

    print ('------ Magic Search Result is ------')
    print (generated_story)
    print ('------------------------------------' + '\n')


'''
    ------ Story Title is ------
    Ice Cream Tasting
    ------ Contrastive Search Result is ------
    My family went to a ice cream shop. We ordered the Ice Cream Truck. It was delicious. The customer service was terrible. We had to leave for another day.
    ------ Magic Search Result is ------
    My family went to a ice cream shop. They ordered three flavors of ice cream. The first one was strawberry, the second was chocolate, and the third was orange. I was excited to try all three flavors. It was very good and I had a great time at the ice cream shop.
    ------------------------------------

    ------ Story Title is ------
    Rainstorm
    ------ Contrastive Search Result is ------
    The weatherman predicted a big storm in the future. He went to his house to check on it. There was nothing to see and it was dark. When he woke up, he realized there was no rain. He decided to stay indoors and watch the weather.
    ------ Magic Search Result is ------
    The rain started to pour down. I heard a rumble in my yard. It was thundering and heavy. My neighbor came over to see what was happening. He had just bought a big umbrella to protect his house.
    ------------------------------------

    ------ Story Title is ------
    Sand Volleyball
    ------ Contrastive Search Result is ------
    I went to the park yesterday. It was raining a lot. I had to use the water pump to get to the park. When I got there, there was nothing to play in the park. I ended up playing volleyball instead.
    ------ Magic Search Result is ------
    I went to the beach with my friends. It was a sand volleyball game. We played for two hours. My friend got to pick his team. He won the game for his team.
    ------------------------------------

    ------ Story Title is ------
    French Braid
    ------ Contrastive Search Result is ------
    The man bought a new scarf. He put it on his head. His wife noticed it was missing. She asked him to look into it. He did not want to look into it.
    ------ Magic Search Result is ------
    I wanted to learn a new style of braid. My friend told me I couldn't afford it. I looked online and found some tutorials. After reading all the tutorials, I decided to go for it. It turns out that the best way to learn new braid is to learn French.
    ------------------------------------

    ------ Story Title is ------
    The Hair Clump
    ------ Contrastive Search Result is ------
    The man shaved his head. He went to get a haircut. His hair fell out. The man had to buy new hair. He was happy that he shaved his head.
    ------ Magic Search Result is ------
    The hair in my hair was a mess. I went to get some shampoo. After shampooing my hair, it looked better. I decided to keep it that way. Now my hair looks great.
    ------------------------------------

    ------ Story Title is ------
    Pig
    ------ Contrastive Search Result is ------
    The man dug a hole. He saw something in the ground. He asked his neighbors for help. His neighbor helped him dig the hole. The man was happy about his contribution.
    ------ Magic Search Result is ------
    The pig came to my door. I put a blanket on it to keep it warm. Then I started playing with it. My neighbor was laughing at me. The pig jumped up and ran away.
    ------------------------------------
'''