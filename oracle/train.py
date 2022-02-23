import os.path

import torch
import argparse
import pandas as pd
import numpy as np
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from cnn import CNN


def parse_opt():
    parser = argparse.ArgumentParser(description='ResNet-MNIST')
    parser.add_argument('--epochs', type=int, default=30, help='input total epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='dataloader batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='optimizer weight_decay')

    args = parser.parse_args()
    return args


def train(epoch):
    total_loss = 0
    total_correct = 0
    total_data = 0
    global iteration

    model.train()
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total_correct += torch.eq(predicted, labels).sum().item()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        total_data += labels.size(0)
        iteration = iteration + 1

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f} iteration:{}".format(epoch + 1,
                                                                               args.epochs,
                                                                               loss,
                                                                               iteration)
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])

    loss = total_loss / len(train_loader)
    acc = 100 * total_correct / total_data
    train_loss.append(loss)
    train_acc.append(acc)

    print('accuracy on train set:%d %%' % acc)


def validate(epoch):
    total_loss = 0
    total_correct = 0
    total_data = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += torch.eq(predicted, labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            total_data += labels.size(0)

            test_bar.desc = "validate epoch[{}/{}]".format(epoch + 1,
                                                           args.epochs)

        loss = total_loss / len(train_loader)
        acc = 100 * total_correct / total_data
        validate_loss.append(loss)
        validate_acc.append(acc)

        print('accuracy on validate set:%d %%\n' % acc)


if __name__ == "__main__":
    args = parse_opt()
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    transform = f.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='../data/', train=True, download=True, transform=transform)
    idx = (train_dataset.targets == 5) | (train_dataset.targets == 7)
    train_dataset.data, train_dataset.targets = train_dataset.data[idx], train_dataset.targets[idx]
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    test_dataset = datasets.MNIST(root='../dataset/', train=False, download=True, transform=transform)
    idx = (test_dataset.targets == 5) | (test_dataset.targets == 7)
    test_dataset.data, test_dataset.targets = test_dataset.data[idx], test_dataset.targets[idx]
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    model = CNN()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_loss = []
    train_acc = []
    validate_loss = []
    validate_acc = []
    iteration = 1

    for i in range(args.epochs):
        train(i)
        validate(i)

    torch.save(model.state_dict(), "./CNN_mnist57.pth".format(device_type))
    epoch = np.arange(1, args.epochs + 1)
    dataframe = pd.DataFrame({'epoch': epoch,
                              'train loss': train_loss,
                              'train accuracy': train_acc,
                              'validate loss': validate_loss,
                              'validate accuracy': validate_acc
                              })
    dataframe.to_csv(r"./loss&acc.csv".format(device_type))
