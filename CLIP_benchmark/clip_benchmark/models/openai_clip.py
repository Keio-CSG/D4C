import torch
import clip
from typing import Union


def load_openai_clip(model_name: str = "ViT-B/32", pretrained: str = None, cache_dir: str = None, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    model, preprocess = clip.load(model_name, device=device)
    tokenizer = clip.tokenize
    return model, preprocess, tokenizer
