# NVIDIA-Certified Associate - Generative AI LLM - Deep-Learning-lab

## Lab 1 - Creating and Training a Neural Network on MNIST dataset
High-level steps:
Import libraries
torch # PyTorch
torch.optim # Optimizer (Adam)
torchvision # Visulaziation for tensors

Import data
import MNIST data and assign it to a variable
Create 2 sets: 1) to train data 2) to validate data
Assign .cuda for GPU processing if available

Transform the data
Convert Python Image Library (PIL) images to tensors using the .ToTensor (see image 1)
These images are black & white so they represent 1st dimension
The next 2 dimensions are for Rows and Columns
Load the data
group the images into batches (32 per batch in this lab)
shuffles the batch for randomizing
iterates - produces batches one by one during training

Creating the model
Flatten image
Create input layer
Assign the no. of output layer neurons
Create a hidden layer
Assign the number of inputs from the input layer
Assign the number of output layer neurons
Assign activation function
Create output layer
assign no. of output neurons
assign activation function
Compiling the model
This is not required, but compiling a model optimizes the code so that the model trains faster

Training the model
Define Loss function
Define Optimizers
Calculate Accuracy
Define how many epochs (Loops of training the entire dataset)
