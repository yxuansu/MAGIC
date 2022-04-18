import torch
import requests
from torch import nn
from PIL import Image

class CLIP(nn.Module):
    def __init__(self, model_name):
        super(CLIP, self).__init__()
        # model name: e.g. openai/clip-vit-base-patch32
        print ('Initializing CLIP model...')
        from transformers import CLIPProcessor, CLIPModel
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.cuda_has_been_checked = False
        print ('CLIP model initialized.')

    def check_cuda(self):
        self.cuda_available = next(self.model.parameters()).is_cuda
        self.device = next(self.model.parameters()).get_device()
        if self.cuda_available:
            print ('Cuda is available.')
            print ('Device is {}'.format(self.device))
        else:
            print ('Cuda is not available.')
            print ('Device is {}'.format(self.device))

    @torch.no_grad()
    def compute_image_representation_from_image_path(self, image_path):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        if self.cuda_available:
            pixel_values = pixel_values.cuda(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds) # [1 x embed_dim]
        return image_embeds

    def compute_image_representation_from_image_instance(self, image):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        if self.cuda_available:
            pixel_values = pixel_values.cuda(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds) # [1 x embed_dim]
        return image_embeds

    def compute_text_representation(self, text_list):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # text_list: a list of text
        text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt",
            max_length=self.tokenizer.max_len_single_sentence + 2, truncation=True)
        # self.tokenizer.max_len_single_sentence + 2 = 77
        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']
        if self.cuda_available:
            input_ids = input_ids.cuda(self.device)
            attention_mask = attention_mask.cuda(self.device)
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = text_outputs[1]
        text_embeds = self.model.text_projection(text_embeds)
        return text_embeds

    def compute_image_text_similarity_via_embeddings(self, image_embeds, text_embeds):
        '''
            image_embeds: 1 x embed_dim
            text_embeds: len(text_list) x embed_dim
        '''
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T
        return logits_per_image.softmax(dim=1) # 1 x len(text_list)

    def compute_image_text_similarity_via_raw_text(self, image_embeds, text_list):
        text_embeds = self.compute_text_representation(text_list)
        return self.compute_image_text_similarity_via_embeddings(image_embeds, text_embeds)

    ### -------------------- functions for building index ---------------------- ###
    def compute_batch_index_image_features(self, image_list):
        '''
            # list of image instances
        '''
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        inputs = self.processor(images=image_list, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        if self.cuda_available:
            pixel_values = pixel_values.cuda(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds) # [1 x embed_dim]
        return image_embeds # len(image_list) x embed_dim

    def compute_batch_index_text_representation(self, text_list):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # text_list: a list of text
        #text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt",
            max_length=self.tokenizer.max_len_single_sentence + 2, truncation=True)
        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']
        if self.cuda_available:
            input_ids = input_ids.cuda(self.device)
            attention_mask = attention_mask.cuda(self.device)
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = text_outputs[1]
        text_embeds = self.model.text_projection(text_embeds)
        logit_scale = self.model.logit_scale.exp()
        text_embeds = text_embeds * logit_scale
        return text_embeds

