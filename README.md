# NVIDIA-Certified Associate - Generative AI LLM - Deep-Learning-labs
These labs are hands-on python labs designed to teach students the fundamentals of Deep Learning by allowing students to build and train very simple Artificial Neural Networks, Transformers and Natural Language Processing


## Lab 1 - Creating and Training a Neural Network on MNIST dataset
This lab is considered the "Hello World" of neural networks.  Before I could follow along and understand what each line of code was doing, I had to spend several hours to understand fundamental concepts associated in this lab:
- Matrices, Scaler vs. Vector vs. Tensor dimensions, y = mx+b (slope intercept) and why it matters in our training
- Python specific topics such as, using PIL for images, argmax, MNIST dataset and why device assignments (CPU vs GPU) can be beneficial when processing tensors
- PyTorch fundamentals: nn.Sequential, nn.Linear, nn.Flatten, 
- What is an Activation function and the popular types (Sigmoid, ReLU, Tanh, Softmax)
- What are Optimizers and the popular types (Adam, Adagrad, RMSprop, SGD)
- Non-mathematical theory of gradient descent and how it helps model adjust the weight in the optimizers
- What is a Loss function and popular types (Mean Squared Error, RootMSE)
- Lab outcome: 
    - Epochs: 5
    - Training - Loss: 62.98, Accuracy:98.89%
    - Validation - Loss: 23.56, Accuracy: 97.98%


## Lab 2 - Creating and Training a Neural Network on American Sign Language dataset
This lab uses American Sign Language(ASL) image dataset to train a model.  The objective is to demonstrate that you may achieve high accuracy with training data, but when running the validation data, the model does not perform well (Overfits).  
- Lab outcome: 
    - Epochs: 20
    - Training - Loss: 11.05, Accuracy:99.66%
    - Validation - Loss: 220.91, Accuracy: 81.64%


## Lab 3 - Creating a simple Convolutional Neural Networks (CNN) for ASL dataset
This lab applies CNN techniques on the ASL model to improve accuracy.
In this lab I learned:
- CNN fundamentals: Regularization, Data Augmentation, Kernel, Stride, Max Pooling, Padding, Dropout
- PyTorch: nn.Conv2d, nn.MaxPool2d, nn.Dropout
- Lab outcome: 
    - Epochs: 20
    - Training - Loss: 00.81, Accuracy:99.66%
    - Validation - Loss: 24.51, Accuracy: 97.49%