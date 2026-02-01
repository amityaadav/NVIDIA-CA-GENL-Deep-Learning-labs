import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Visualization tools
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

# set device to GPU (cuda) if available else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# download MNIST dataset
train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)

# dataset
print(train_set)
print(valid_set)

x_0, y_0 = train_set[0]

# dispay the first PIL Image item in the dataset
# x_0.show()

# display the first item's validation value
print(f"The actual value of first PIL image is: {y_0}")

# convert the image to a tendor
trans = transforms.Compose([transforms.ToTensor()])

x_0_tensor = trans(x_0)

# view the size of each tensor dimension (Color x Height X Width)
print(f"The size of each demension is: {x_0_tensor.size()}")

# view the tensor object
print(x_0_tensor)

print(f"by default the tensor is run on: {x_0_tensor.device}")

# assign tensor object to NVIDIA GPU (if available)
# x_0_gpu = x_0_tensor.cuda()

# you can convert a tensor back to PIL image by using to_pil_image
# image = F.to_pil_image(x_0_tensor)
# plt.imshow(image, cmap='gray')

# print(image.show())

# we can apply our list of transforms to a dataset. One such way is to set it to a dataset's transform variable.
train_set.transform = trans
valid_set.transform = trans

# load data into batch of 32 images / batch
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

# here's a sample of what the first batch looks like.
for batch in train_loader:
    inputs, labels = batch
    print(f"Batch shape: {inputs.shape}")
    print(f"Labels: {labels}")
    break  # Stop after first batch

input_size = 1 * 28 * 28
n_classes = 10

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512), # Input Layer
    nn.ReLU(),                      # Input layer activation function
    nn.Linear(512,512),         # Hidden Layer
    nn.ReLU(),                      # Input layer activation function
    nn.Linear(512, n_classes)   # Output layer
]
print(f"These are the model lyear properties: {layers}")