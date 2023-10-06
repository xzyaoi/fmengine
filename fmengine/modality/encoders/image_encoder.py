import torch
import numpy as np

import open_clip

class ImageEncoder():
    def __init__(self, pretrained_name: str) -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(pretrained_name)
        self.model.eval()
    
    def encode(self, images):
        with torch.inference_mode():
            images = [self.preprocess(image) for image in images]
            image_input = torch.tensor(np.stack(images))
            return self.model.encode_image(image_input).float()