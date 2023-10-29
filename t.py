import torch
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn
import torch.optim as optim

import pandas as pd
import numpy as np

from torchvision import transforms


class CustomMNIST(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file).values
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx, 1:].reshape(28, 28).astype(np.uint8)
        label = torch.tensor(int(self.data[idx, 0]))

        if self.transform:
            image = self.transform(image)

        return image, label
    
    


transform = transforms.Compose([transforms.ToTensor()])
dt = CustomMNIST('train.csv', transform=transform)
dl = DataLoader(dt, batch_size=64, shuffle=True)


net = nn.Sequential(
    nn.Flatten(),  
    nn.Linear(784, 128),  
    nn.ReLU(), 
    nn.Linear(128, 10)  
)


lf = nn.CrossEntropyLoss()
opt = optim.SGD(net.parameters(), lr=0.01)


num_epoch = 10

for epoch in range(num_epoch):
    net.train()
    running_loss = 0.0

    for bd, bl in dl:
        opt.zero_grad()
        outputs = net(bd)
        loss = lf(outputs, bl)
        loss.backward()
        opt.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch + 1}, Loss: {running_loss/ len(dl)}')
        


test_dt = CustomMNIST('test.csv', transform=transform)
test_dl = DataLoader(test_dt, batch_size=64, shuffle=False)


net.eval()  
correct_predictions = 0
total_samples = 0

with torch.no_grad():  
    for bd, bl in test_dl:
        outputs = net(bd)
        _, predicted = torch.max(outputs, 1) 
        correct_predictions += (predicted == bl).sum().item()
        total_samples += bl.size(0)

accuracy = correct_predictions / total_samples
print(f'Test Accuracy: {accuracy * 100:.2f}%')


