import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import pickle
import random
from models.LinearModel import KronLinear


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
        self.kronlinear1 = KronLinear(784, 10, patchsize=(49, 2), structured_sparse=True, rank=20)
        self.kronlinear2 = KronLinear(784, 10, patchsize=(2, 10), structured_sparse=True, rank=20)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = x.view(-1, 784)
        return self.activation(self.kronlinear1(x)), self.activation(self.kronlinear2(x))



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
                output1, output2 = model(x)
                loss = F.cross_entropy(output1, y)
                loss += F.cross_entropy(output2, y)
                loss += torch.norm(model.kronlinear1.s, p=2) * 100 / torch.numel(model.kronlinear1.s)
                loss += torch.norm(model.kronlinear2.s, p=2) * 100 / torch.numel(model.kronlinear2.s)
                
                loss.backward()
                optimizer.step()
                # if i % 100 == 0:
                #     print(f'iteration {i}, loss {loss.item()}')
        
        with torch.no_grad():
            correct1 = 0
            correct2 = 0
            total = 0
            weights.append(model.kronlinear1.s)
            weights.append(model.kronlinear2.s)
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                output1, output2 = model(x)
                _, predicted1 = torch.max(output1, 1)
                _, predicted2 = torch.max(output2, 1)
                total += y.size(0)
                correct1 += (predicted1 == y).sum().item()
                correct2 += (predicted2 == y).sum().item()
            print(f'kronlinear 1 fold {fold} accuracy: {correct1/total}')
            print(f'kronlinear 2 fold {fold} accuracy: {correct2/total}')
            accuracy.append(correct1/total)
            accuracy.append(correct2/total)
        total_params = torch.numel(model.kronlinear1.s)  + torch.numel(model.kronlinear2.s)
        zero_params = torch.sum(model.kronlinear1.s < 1e-5) + torch.sum(model.kronlinear2.s < 1e-5)
        
        print('linear1 zero params:', torch.sum(model.kronlinear1.s < 1e-5), 'linear1 s numels:', torch.numel(model.kronlinear1.s))
        print(torch.norm(model.kronlinear1.s, p=2).detach().cpu().numpy() )
        print('linear2 zero params:', torch.sum(model.kronlinear2.s < 1e-5), 'linear2 s numels:', torch.numel(model.kronlinear2.s))
        print(torch.norm(model.kronlinear2.s, p=2).detach().cpu().numpy() )
        # zero_params to int
        zero_params = zero_params.item()
        sparsity = zero_params / total_params
        sparse.append(sparsity)
        #save weights and sparsity array
        with open(f'./results/one_layer_KPD_{fold}.pkl', 'wb') as f:
            f.write(pickle.dumps((weights, sparse)))
        torch.save(model.state_dict(), f'./results/one_layer_KPD_{fold}.pth')
    import numpy as np
    accuracy = np.array(accuracy)
    print(np.mean(accuracy[0::2]), np.std(accuracy[0::2]))
    print(np.mean(accuracy[1::2]), np.std(accuracy[1::2]))
        
    print(np.mean(sparse), np.std(sparse))
    return accuracy

model = OneLayer()

train_test(model)
