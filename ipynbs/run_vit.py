import os
import sys
from pathlib import Path
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

sys.path.append(str(Path.cwd().parent))
from timm.models.vision_transformer import VisionTransformer
from timm import create_model

model = create_model("vit_base_patch16_224",pretrained=True)

img = read_image('dog_cat.jpg')
img = Resize((224,224))(img)
img = img/255.0

y, attn = model(img.unsqueeze(dim=0))
print(y.shape)