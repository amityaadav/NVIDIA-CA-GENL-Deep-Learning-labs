# NVIDIA Deep Learning Labs
These labs are hands-on python labs designed to teach the fundamentals of Deep Learning by allowing students to build and train very simple Artificial Neural Networks. Additional labs discuss Transformer architecture and Natural Language Processing


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
    - Training - Loss: 62.98, Accuracy: 98.89%
    - Validation - Loss: 23.56, Accuracy: 97.98%


## Lab 2 - Creating and Training a Neural Network on American Sign Language dataset
This lab uses American Sign Language(ASL) image dataset to train a model.  The objective is to demonstrate that you may achieve high accuracy with training data, but when running the validation data, the model does not perform well (Overfits).  
- Lab outcome: 
    - Epochs: 20
    - Training - Loss: 11.05, Accuracy: 99.66%
    - Validation - Loss: 220.91, Accuracy: 81.64%


## Lab 3 - Creating a simple Convolutional Neural Networks (CNN) for ASL dataset
In lab 2, the neural network could not predict ASL images higher than 81% accuracy.  While the loss for training dataset remained low, the loss continued to stay high for validation dataset over the last few epochs indicating to model overfitting.  In Lab 3, CNN techniques were applied on the ASL dataset to improve accuracy.

In this lab I learned:
- CNN fundamentals: Regularization, Data Augmentation, Kernel, Stride, Pooling(Average/Max), Padding, Dropout
- PyTorch: nn.Conv2d, nn.MaxPool2d, nn.Dropout
- Lab outcome: 
    - Epochs: 20
    - Training - Loss: 00.81, Accuracy: 99.66%
    - Validation - Loss: 24.51, Accuracy: 97.49%


## Lab 4.A - Using the prior CNN model and augmenting the data
The objective of this lab was to demonstrate the importance of data augmentation and what factors/features to consider when augmenting data.  Data augmentation reduced the loss and increased the accuracy of the model.  Several augmenting techniques were applied to the images, such as increased number of quantiy, random rotation, slightly dialing in brightness and contrast, which provides the model with variance in patterns.  Image augmentation should always consider the use case for what it's being trained for, for example, an inverse ASL image is not realistic while slightly rotated/tilted are ok.
- Lab outcome: 
    - Epochs: 20
    - Training - Loss: 12.28 Accuracy: 99.59%
    - Validation - Loss: 7.49 Accuracy: 98.76%


## Lab 4.B - Use the saved CNN model from Lab 4.A, load an unseen image and run it through the model
The objective in this lab was to load an image that the model has never seen and inference it through the trained model from previous lab. 


## Lab 5 - Demonstrate transfer learning; Leverage a pre-trained model to inference unseen data
The objective of this lab was to demonstrate the use of transfer learning by using a pre-trained model called VGG16: (https://www.kaggle.com/datasets/crawford/vgg16).  The model is trained on very large dataset with an output of 1000 distinct categories of prediction (animals, objects, etc.).  In this lab, the model is used to inference unseen images of animals and verify that the predictions are correct. 


## Lab 6 - Natural Language Processing
The objective of this lab was to tokenize text and use embeddings.  Importing BERT model and passing a statement to tokenize text.


## Lab 7 - Assessment lab
The objective in this lab was to classify whether an image is fresh{apples, bananas, oranges} or rotten{apples, bananas, oranges} with 92% or greater accuracy.  VGG16 dataset is used for transfer learning, the model is frozen and trained on fruits data first, then unfrozen and trained on the entire dataset.  The 92% threshold forces you to try various methods to mitigate overfitting. 

Attempt 1 - Baseline with no data augmentation.
- Freeze (10 Epoch)
Train - Loss: 4.6634 Accuracy: 0.9567
Valid - Loss: 3.4179 Accuracy: 0.8906
- Unfreeze (1 Epoch)
Train - Loss: 18.7144 Accuracy: 0.8533
Valid - Loss: 12.5195 Accuracy: 0.7964
- Final accuracy: 79.64%


Attempt 2 - increase the number of epochs for frozen model to 12, and increase epochs for unfrozen model to 2.  Add data augmentation.
- Freeze (12 Epoch)
Train - Loss: 4.5569 Accuracy: 0.9658
Valid - Loss: 1.5734 Accuracy: 0.9757
- Unfreeze (2 Epoch)
Train - Loss: 13.9137 Accuracy: 0.9013
Valid - Loss: 9.7231 Accuracy: 0.8871
- Final accuracy: 90.13%


Attempt 3 - Add dropout for increased regularization, update to a higher learning rate, and add aditional data augmentation.
- Freeze (15 Epoch)
Train - Loss: 7.2096 Accuracy: 0.9237
Valid - Loss: 3.3623 Accuracy: 0.9058
- Unfreeze (5 Epoch)
Train - Loss: 4.6075 Accuracy: 0.9567
Valid - Loss: 3.1694 Accuracy: 0.9301
- Final accuracy: 93.01%