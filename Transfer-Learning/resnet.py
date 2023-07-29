import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
epoch = 2
batch_size = 64
learning_reate = 1e-4  # 0.0001

transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),  # 0 ~  1 Normalize
     transforms.Normalize( (0.5,0.5,0.5), (0.5, 0.5, 0.5))  # -1 ~ 1
     ])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)