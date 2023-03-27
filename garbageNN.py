import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datasetGarbage import datasetGarbage


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(196608, 30)
        self.fc2 = nn.Linear(30, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def save_model(model_nn):
    with open('save_model.pkl', "wb") as f:
        pickle.dump(model_nn, f)
        f.close()
        print("The model is saved")


def load_model(model_nn):
    with open('save_model.pkl', "rb") as f:
        model_nn = pickle.load(f)
    return model_nn


def progressbar(prcnt):
    progress = "|"
    for i in range(100):
        if i <= prcnt:
            progress += "="
        else:
            progress += '-'
    progress += "|"
    return progress


def check_accuracy(loader, model):
    device_C = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Checking accuracy:")
    # if loader.dataset.train:
    #    print("Checking accuracy on training data")
    # else:
    #    print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device_C)
            y = y.to(device=device_C)
            scores = model(x).to(device=device_C)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channel = 3
    num_classes = 6
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 8

    dataset = datasetGarbage(csv_file='garbage.csv', root_dir='garbage-resized', transform=transforms.ToTensor())
    train_set, test_set = torch.utils.data.random_split(dataset, [2000, 526])

    # train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    model = CNN(in_channels=in_channel, num_classes=num_classes).to(device)

    print("load model? y/n")
    if str(input()) == 'y':
        model = load_model(model)
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    progress = 0
    length_process = float(train_loader.__len__() * num_epochs)
    print(f'\rProgress: {float(progress) / length_process * 100:.2f}% ' + progressbar(0))
    for epoch in range(num_epochs):
        check_accuracy(train_loader, model)
        check_accuracy(test_loader, model)
        for batch_idx, (data, targets) in enumerate(train_loader):  # train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # data = data.reshape(data.shape[0], -1)
            model.train()
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress += 1
            print(f'\rProgress: {float(progress) / length_process * 100:.2f}% ' + progressbar(
                float(progress) / length_process * 100), end='')
        print("")
        save_model(model)
    check_accuracy(train_loader, model)  # ,train_loader, model)
    check_accuracy(test_loader, model)  # , model)
