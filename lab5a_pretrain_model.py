import torch
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the VGG16 network *pre-trained* on the ImageNet dataset
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

pre_trans = weights.transforms()

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.show()          # matplot function to show the imapge called using show_image()

#show_image("data/doggy_door_images/happy_dog.jpg")

# Pre-process the image
def load_and_process_image(file_path):
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(file_path).shape)

    image = tv_io.read_image(file_path).to(device)
    image = pre_trans(image)  # weights.transforms()
    image = image.unsqueeze(0)  # Turn into a batch
    return image

# Function to classify dogs, cats and everything else
import numpy as np

def doggy_door(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    idx = model(image).argmax(dim=1).item()
    print("Predicted index:", idx)
    if 151 <= idx <= 268:
        print("Doggy come on in!")
    elif 281 <= idx <= 285:
        print("Kitty stay inside!")
    else:
        print("You're not a dog! Stay outside!")

# Test the predictions
doggy_door("data/doggy_door_images/brown_bear.jpg")
#doggy_door("data/doggy_door_images/happy_dog.jpg")
#doggy_door("data/doggy_door_images/sleepy_cat.jpg")