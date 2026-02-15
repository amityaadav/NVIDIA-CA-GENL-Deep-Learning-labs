import torch.nn as nn
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# Set device to GPU (cuda) if available else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# Load ASL dataset from csv file
train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")

# Print first 5 rows from the train file
print(f"Training dataset: {train_df.head()}")
print(f"Training dataset: {train_df.head()}")

# Extract the label column
y_train = train_df.pop('label')
y_valid = valid_df.pop('label')

# Preview of labels
print(y_train)

# Extract the image data from the spreadsheet
x_train = train_df.values
x_valid = valid_df.values

print(f"Shape of training data: {x_train.shape}")
print(f"Shape of training data: {y_train.shape}")
print(f"Shape of validation data: {x_valid.shape}")
print(f"Shape of validation data: {y_valid.shape}")


plt.figure(figsize=(40,40))

#loop to visualize the data
num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]

    image = row.reshape(28,28)
    plt.subplot(1, num_images, i+1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')

x_train = train_df.values / 255
x_valid = valid_df.values / 255

# custom dataset
class MyDataset(Dataset):
    def __init__(self, x_df, y_df):
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)

BATCH_SIZE = 32

train_data = MyDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_N = len(train_loader.dataset)

valid_data = MyDataset(x_valid, y_valid)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
valid_N = len(valid_loader.dataset)

batch = next(iter(train_loader))

# define the model
input_size = 28 * 28
n_classes = 26

# Sequential model expects sequential arguments to be passed 
model = nn.Sequential()
print(f"Sequence of the model: {model}")

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU(),  # Activation for hidden
    nn.Linear(512, n_classes)  # Output
)

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# train function
def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

# validation function
def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

# training loop
epochs = 20

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()