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

x_0, y_0 = train_set[0]     # Primary test data
x_7, y_7 = train_set[6]     # Secondary test data

## display the first PIL Image item in the dataset
# x_0.show()

# display the first item's validation value
print(f"The actual value of first PIL image is: {y_0}")

# convert the image to a tensor
trans = transforms.Compose([transforms.ToTensor()])

x_0_tensor = trans(x_0)

# view the size of each tensor dimension (Color x Height X Width)
print(f"The size of each demension is: {x_0_tensor.size()}")

# view the tensor object
print(x_0_tensor)

print(f"by default the tensor is run on: {x_0_tensor.device}")

## assign tensor object to NVIDIA GPU (if available)
# x_0_gpu = x_0_tensor.cuda()

## you can convert a tensor back to PIL image by using to_pil_image
# image = F.to_pil_image(x_0_tensor)
# plt.imshow(image, cmap='gray')

## print(image.show())

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


# Define inputer,hidden, and output layers.  Define activation functions
input_size = 1 * 28 * 28
n_classes = 10

layers = [
    nn.Flatten(),               # Flatten all layers so that we turn n-dimensional data into a vector, 
                                # because we are using "Linear"/fully-connected layers in this lab, 
                                # thus it's a mathematical dependency, not a requirement
    nn.Linear(input_size, 512), # Input Layer
    nn.ReLU(),                      # Input layer activation function
    nn.Linear(512,512),         # Hidden Layer
    nn.ReLU(),                      # Input layer activation function
    nn.Linear(512, n_classes)   # Output layer
]
print(f"These are the layer properties: {layers}")

# Sequential model expects sequential arguments to be passed 
model = nn.Sequential(*layers)
print(f"Sequence of the model: {model}")

# Define Loss function (grade a model for guessing correctly)
loss_function = nn.CrossEntropyLoss()

# Optimizer (how to learn from the grade, to do better next time)
optimizer = Adam(model.parameters())

# Calculating accuracy.  Total no. of prediction is the size of dataset.  
train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

# Calculate the accuracy of each batch
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

# Function to train the model
def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

# Validation function
def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

# Training loop
epochs = 5

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

# Prediction - convert x_0/x_7 object to tensor and add batch dimension
x_7_tensor = trans(x_7)  # Convert PIL to tensor: [1, 28, 28]
x_7_batch = x_7_tensor.unsqueeze(0)  # Add batch dim: [1, 1, 28, 28]
x_7_batch = x_7_batch.to(device)  # Move to same device as model

prediction = model(x_7_batch)
print(f"Prediction Tensor: {prediction}")
print(f"Predicted class: {prediction.argmax(dim=1).item()}")
print(f"Actual: {y_7}")