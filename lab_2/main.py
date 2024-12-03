import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import random


class dataset(torch.utils.data.Dataset):
    def __init__(self,file_list,transform=None):
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        label = img_path.split("\\")[1].split("_")[0]
        if label == 'leopard':
            label=1
        elif label == 'tiger':
            label=0
            
        return img_transformed,label


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def data_loader(batch_size: int) -> torch.utils.data.DataLoader:

    "creating data loaders for model training"

    train_list, test_list, validate_list = return_lists()
    train_transforms, test_transforms, validate_transforms = data_augmentation()

    train_data = dataset(train_list, transform=train_transforms)
    test_data = dataset(test_list, transform=test_transforms)
    validate_data = dataset(validate_list, transform=validate_transforms)

    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset = validate_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, validate_loader
    

def data_augmentation() -> transforms.transforms.Compose:
    "Changes image parameters"
    train_transforms =  transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(240),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    validate_transforms = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.RandomResizedCrop(240),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])


    test_transforms = transforms.Compose([   
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(240),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
    
    
    return train_transforms, validate_transforms, test_transforms


def return_lists() -> list:
    
    "the function returns an array containing path to the images (split datas)"
    from sklearn.model_selection import train_test_split

    train_dir = "train_dir" # path to dir with images

    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    train_list, temp_list = train_test_split(train_list, test_size=0.2)
    test_list, validate_list = train_test_split(temp_list, test_size=0.5)

    # print(train_list[0].split("\\")[1].split("_")[0])

    return train_list, test_list, validate_list


def main() -> None:
    device = "cpu"
    lr = 0.001 # learning_rate
    batch_size = 100 # we will use mini-batch method
    epochs = 10 # How much to train a model

    train_loader, test_loader, validate_loader = data_loader(batch_size)
    model = Cnn().to(device)
    
    optimizer = optim.Adam(params = model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = ((output.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc/len(train_loader)
            epoch_loss += loss/len(train_loader)

            train_accuracies.append(epoch_accuracy)
            train_losses.append(epoch_loss)

            
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        
        
        with torch.no_grad():
            epoch_val_accuracy=0
            epoch_val_loss =0
            for data, label in validate_loader:
                data = data.to(device)
                label = label.to(device)
                
                val_output = model(data)
                val_loss = criterion(val_output,label)
                
                
                acc = ((val_output.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc/ len(validate_loader)
                epoch_val_loss += val_loss/ len(validate_loader)
            
            val_accuracies.append(epoch_val_accuracy)
            val_losses.append(epoch_val_loss)

            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))
    

    draw_graph(train_losses, val_losses, 'Loss', 'Epochs', 'Loss Value')
    draw_graph(train_accuracies, val_accuracies, 'Accuracy', 'Epochs', 'Accuracy Value')

    leopard_probs = []
    model.eval()
    with torch.no_grad():
        for data, fileid in test_loader:
            data = data.to(device)
            preds = model(data)
            preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
            leopard_probs += list(zip(list(fileid), preds_list))
    leopard_probs.sort(key = lambda x : int(x[0]))

    image_test_probs(leopard_probs)

    torch.save(model.state_dict(), 'trained_model.pth')


def image_test_probs(leopard_probs: []) -> None:
    test_list = return_lists()[1]

    idx = list(map(lambda x: x[0],leopard_probs))
    prob = list(map(lambda x: x[1],leopard_probs))

    submission = pd.DataFrame({'id':idx,'label':prob})
    
    test_list = return_lists()[1]

    id_list = []
    class_ = {0: 'tiger', 1: 'leopard'}

    fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')

    for ax in axes.ravel():
        i = random.choice(submission.index.values)
        print(f"Selected index: {i}")

        label = submission.loc[submission.index == i, 'label'].values[0]
        
        print(f"Label = {label}")
  
        if label > 0.5:
            label = 1
        else:
            label = 0
            
        img_path = os.path.join(test_list[i])
        print(f"Image path: {img_path}")
        
        img = Image.open(img_path)
        ax.set_title(class_[label])
        ax.imshow(img)

    plt.show()


def image_recognition() -> None:
    "Test model"
    model = Cnn().to('cpu')

    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    image_path = "D:\\python_labs\\datas\\leopard\\leopard_1039.jpg"


    # Определение преобразований для входного изображения
    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(240),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


    image = Image.open(image_path)
    tensor_image = transform(image).unsqueeze(0)


    with torch.no_grad():
        output = model(tensor_image)

    probabilities = F.softmax(output, dim=1)

    print("Предсказанные вероятности:", probabilities)
    predicted_class = torch.argmax(probabilities).item()
    if predicted_class == 1:
        print("Предсказанный класс: leopard")
    else:
        print("Предсказанный класс: tiger")


def draw_graph(train_values : [], val_values : [], title : str, xlabel : str, ylabel : str) -> None:
    "Draw graphics "
    train_values = [val.detach().numpy() for val in train_values]
    val_values = [val.detach().numpy() for val in val_values]

    plt.plot(train_values, label='Train')
    plt.plot(val_values, label='Validation')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    image_recognition()