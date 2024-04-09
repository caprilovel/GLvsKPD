import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import pickle
import random

from utils.loss import  group_pattern
#using 5-fold of mnist
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

full_dataset = torch.utils.data.ConcatDataset([train_mnist, test_mnist])
kf = KFold(n_splits=5, shuffle=True)
batch_size = 64


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()
        self.linear = nn.Linear(784, 10)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = x.view(-1, 784)
        return self.activation(self.linear(x))


# calculate the parameter and flops 
model = OneLayer()

def train_test(model, ):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lasso_lambda = 0.0001

    accuracy = []
    sparse = []
    weights = []
    for fold, (train_index, test_index) in enumerate(kf.split(full_dataset)):
        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_dataset, train_index), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_dataset, test_index), batch_size=batch_size, shuffle=True)
        
        for epoch in range(50):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output, y)
                l2_regularization = torch.norm(group_pattern(10, 2, mat=model.linear.weight),p=2, dim=1)
                l1_regularization = torch.norm(l2_regularization, p=1)
                loss += lasso_lambda * l1_regularization

                loss.backward()
                optimizer.step()

        
        with torch.no_grad():
            correct = 0
            total = 0
            weights.append(model.linear.weight)
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(f'fold {fold} accuracy: {correct/total}')
            accuracy.append(correct/total)
        total_params = torch.numel(model.linear.weight) 
        zero_params = torch.sum(model.linear.weight < 1e-5)
        # zero_params to int
        zero_params = zero_params.item()
        sparsity = zero_params / total_params
        sparse.append(sparsity)
        #save weights and sparsity array
        with open(f'./results/one_layer_group_lasso_{fold}.pth', 'wb') as f:
            f.write(pickle.dumps((weights, sparse)))
    import numpy as np
    accuracy = np.array(accuracy)
    print(np.mean(accuracy), np.std(accuracy))
    
    print(np.mean(sparse), np.std(sparse))
    return accuracy

model = OneLayer()

train_test(model)
