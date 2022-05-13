"""
download checkpoints cambridgeltl/simctg_rocstories, cambridgeltl/magic_mscoco and openai/clip-vit-base-patch32
from https://huggingface.co for faster inference of the web demo
"""
import sys

sys.path.append(r"./story_generation/language_model")
from typing import Optional
from PIL import Image
import torch
from transformers import AutoTokenizer
from cog import BasePredictor, Path, Input, BaseModel

from image_captioning.language_model.simctg import SimCTG as ImageCaptioningSimCTG
from story_generation.language_model.simctg import SimCTG as StoryGenerationSimCTG
from image_captioning.clip.clip import CLIP as ImageCaptioningCLIP
from story_generation.clip.clip import CLIP as StoryGenerationCLIP


class Output(BaseModel):
    image_caption: Optional[str]
    contrastive_search_result: Optional[str]
    magic_search_result: Optional[str]


class Predictor(BasePredictor):
    def setup(self):

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        # Load Language Model
        magic_mscoco = "checkpoints/magic_mscoco"
        sos_token, pad_token = r"<-start_of_text->", r"<-pad->"

        simctg_rocstories = "checkpoints/simctg_rocstories"
        self.simctg_rocstories_tokenizer = AutoTokenizer.from_pretrained(
            simctg_rocstories
        )

        self.image_captioning_model = ImageCaptioningSimCTG(
            magic_mscoco, sos_token, pad_token
        ).to(self.device)
        self.story_generation_model = StoryGenerationSimCTG(
            simctg_rocstories, self.simctg_rocstories_tokenizer.pad_token_id
        ).to(self.device)
        self.image_captioning_model.eval()
        self.story_generation_model.eval()

        start_token = self.image_captioning_model.tokenizer.tokenize(sos_token)
        start_token_id = self.image_captioning_model.tokenizer.convert_tokens_to_ids(
            start_token
        )
        self.input_ids = torch.LongTensor(start_token_id).view(1, -1).to(self.device)

        model_name = "checkpoints/clip-vit-base-patch32"
        self.image_captioning_clip = ImageCaptioningCLIP(model_name).to(self.device)
        self.story_generation_clip = StoryGenerationCLIP(model_name).to(self.device)
        self.image_captioning_clip.eval()
        self.story_generation_clip.eval()

    def predict(
        self,
        task: str = Input(
            choices=["Image Captioning", "Story Generation"],
            default="Image Captioning",
            description="Choose a task.",
        ),
        image: Path = Input(
            description="Input image.",
        ),
        title: str = Input(
            default=None,
            description="Title for story generation (not used for Image Captioning). "
            "5 sentences will be generated for a story.",
        ),
    ) -> Output:

        if task == "Image Captioning":
            generation_model = self.image_captioning_model
            clip = self.image_captioning_clip
            k, alpha, beta, decoding_len = 45, 0.1, 2.0, 16
            image = Image.open(str(image))

            output = generation_model.magic_search(
                self.input_ids, k, alpha, decoding_len, beta, image, clip, 60
            )
            return Output(image_caption=output)

        else:
            assert title is not None, "Please provide title for story generation."
            title = title.rstrip() + " <|endoftext|>"
            generation_model = self.story_generation_model
            clip = self.story_generation_clip

            k, alpha, beta, decoding_len = 5, 0.6, 0.15, 100
            eos_token = r"<|endoftext|>"

            title_tokens = self.simctg_rocstories_tokenizer.tokenize(title)
            title_id_list = self.simctg_rocstories_tokenizer.convert_tokens_to_ids(
                title_tokens
            )
            title_ids = torch.LongTensor(title_id_list).view(1, -1).to(self.device)

            contrastive_output, _ = generation_model.fast_contrastive_search(
                title_ids, k, alpha, decoding_len, eos_token
            )
            _, contrastive_generated_story = generation_model.parse_generated_result(
                contrastive_output, num_of_sentences_to_keep=5
            )

            image = Image.open(str(image))
            magic_output, _ = generation_model.magic_search(
                title_ids,
                k,
                alpha,
                decoding_len,
                beta,
                image,
                clip,
                60,
                eos_token,
            )
            _, magic_generated_story = generation_model.parse_generated_result(
                magic_output, num_of_sentences_to_keep=5
            )

            return Output(
                contrastive_search_result=contrastive_generated_story,
                magic_search_result=magic_generated_story,
            )
