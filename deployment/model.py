import numpy as np
import torch
from torchvision import transforms
import backbones
import simplenet
from logger import logger
import PIL.Image as Image
from typing import Union
import matplotlib.pyplot as plt

device : torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

model = simplenet.SimpleNet(device)
backbone = backbones.load('wideresnet50')
model.load(
    backbone=backbone,
    layers_to_extract_from=['layer2', 'layer3'],
    device=device,
    input_shape=(3, 224, 224),
    pretrain_embed_dimension=1536,
    target_embed_dimension=1536,
    meta_epochs=10,
    aed_meta_epochs=1,
    gan_epochs=4
)
model.set_model_dir('models', 'MVTEC')
model.load_model()

def inference(image_path) -> tuple[bool, Union[str, None]]:
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform_img(image)
    result = model.predict(transformed_image.unsqueeze(0))
    logger.info(f"Result: {torch.nn.functional.sigmoid(torch.tensor(result[0]))}")
    isDefect = torch.nn.functional.sigmoid(torch.tensor(result[0])) > 0.5
    if isDefect:
        fig, ax = plt.subplots()
        ax.imshow(image.resize((224, 224)))
        ax.imshow(result[1][0], cmap='inferno', alpha=0.7)
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.savefig('temps/overlaid_image.png', dpi=300, bbox_inches='tight', pad_inches=0)
        return True, 'temps/overlaid_image.png'
    return False, None