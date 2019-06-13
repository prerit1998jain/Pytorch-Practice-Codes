import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

# Device Configuration
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameter Tuning
batch_size = 100
num_classes = 10
num_epochs = 5
learning_rate = 0.001

# Data Downloading and Building DataLoader
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor())

trainLoader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
testLoader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False )


# Defining the model
class CNN(nn.Module):
    def __init__(self, num_classes = 10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size = 5, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size = 5, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Defining an instance of the class
model = CNN(num_classes = 10).to(device)

# Defining the Loss Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(), lr = learning_rate)

# Training the model
total_step = len(trainLoader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainLoader):
        images = images.to(device)
        labels = labels.to(device)

        # Performing the Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print("epoch: [{}/{}], step: [{}/{}], Loss: {:.4f} \n".format(epochs, num_epochs, i+1, total_step, loss.item()))
