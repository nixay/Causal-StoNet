import torch
import numpy as np
import torch.nn as nn
import os
import errno
from network_replicate import Net
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

########################################################################################################################
# network parameters
num_hidden = 3
hidden_dim = [64, 32, 16]
input_dim = 5

# task
regression_flag = True

# Generate data using StoNet
if regression_flag is True:
    net_sim = Net(num_hidden, hidden_dim, input_dim, 1)
    para_sim = list(net_sim.parameters())
else:
    net_sim = Net(num_hidden, hidden_dim, input_dim, 2)
    para_sim = list(net_sim.parameters())

class SimStoNet_Cont(Dataset):
    """
    generate simulation data using StoNet for regression task

    seed: float
        random seed
    size: float
        size of the simulation dataset
    """
    def __init__(self, seed, size):
        self.seed = seed
        self.size = size
        self.x, self.y = [], []
        sigma = 1.0
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        for i in range(int(self.size)):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp = np.repeat(ee, 5)
            for j in range(5):
                x_temp[j] += np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp /= np.sqrt(2)
            x_temp = torch.FloatTensor(x_temp)

            y_temp = net_sim(x_temp).item()
            y_temp = np.reshape(y_temp, 1)
            y_temp = torch.FloatTensor(y_temp)

            self.x.append(x_temp)
            self.y.append(y_temp)

    def __len__(self):
        return int(self.size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        return y, x


class SimStoNet_Bin(Dataset):
    """
    generate simulation data using StoNet for binary classification task

    seed: float
        random seed
    size: float
        size of the simulation dataset
    """
    def __init__(self, seed, size):
        self.seed = seed
        self.size = size
        self.x, self.label = [], []
        sigma = 1.0
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        for i in range(int(self.size)):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp = np.repeat(ee, 5)
            for j in range(5):
                x_temp[j] += np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp /= np.sqrt(2)
            x_temp = torch.FloatTensor(x_temp)

            prob = net_sim(x_temp).detach()
            label_temp = prob.argmax()
            label_temp = torch.LongTensor(label_temp)

            self.x.append(x_temp)
            self.label.append(label_temp)

    def __len__(self):
        return int(self.size)

    def __getitem__(self, idx):
        label = self.label[idx]
        x = self.x[idx]
        return label, x

########################################################################################################################
# imputation parameters
impute_lrs = [1e-7, 1e-7, 1e-7]
ita = 0.5
mh_step = 1
sigma_list = [1e-6, 1e-5, 1e-4, 1e-3]

# training parameters
batch_size = 50
para_momentum = 0.9
para_lrs = [1e-4, 1e-5, 1e-6, 1e-7]
epochs = 1000

# loss function
if regression_flag is True:
    loss_func_sum = nn.MSELoss(reduction='sum')
    loss = nn.MSELoss()
    train_loss_path = np.zeros(epochs)
else:
    loss_func_sum = nn.CrossEntropyLoss(reduction='sum')
    loss = nn.CrossEntropyLoss()
    train_loss_path = np.zeros(epochs)
    train_accuracy_path = np.zeros(epochs)
sse = nn.MSELoss(reduction='sum')

# simulation dataset
if regression_flag is True:
    train_set = SimStoNet_Cont(3, 500)
else:
    train_set = SimStoNet_Bin(2, 500)

train_data = DataLoader(train_set, batch_size)

# network
if regression_flag is True:
    output_dim = 1
else:
    output_dim = 2

net = Net(num_hidden, hidden_dim, input_dim, output_dim)
# net.state_dict()['0.weight'][:] = para_sim[0]
# net.state_dict()['0.bias'][:] = para_sim[1]
# net.state_dict()['1.1.weight'][:] = para_sim[2]
# net.state_dict()['1.1.bias'][:] = para_sim[3]
# net.state_dict()['2.1.weight'][:] = para_sim[4]
# net.state_dict()['2.1.bias'][:] = para_sim[5]
# net.state_dict()['3.1.weight'][:] = para_sim[6]
# net.state_dict()['3.1.bias'][:] = para_sim[7]
para = list(net.parameters())
print(para[1])
print(para[3])
print(para[5])
print(para[7])
# path to save the result
if regression_flag is True:
    base_path = os.path.join('.', 'stonet', 'result', 'sim', 'regression')
else:
    base_path = os.path.join('.', 'stonet', 'result', 'sim', 'classification')
spec = str(impute_lrs) + '_' + str(para_lrs) + '_' + str(hidden_dim) + '_' + str(epochs)
PATH = os.path.join(base_path, spec)

if not os.path.isdir(PATH):
    try:
        os.makedirs(PATH)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(PATH):
            pass
        else:
            raise

# define optimizer
optimizer_list = []
for i in range(num_hidden + 1):
    optimizer_list.append(SGD(net.module_dict[str(i)].parameters(), lr=para_lrs[i], momentum=para_momentum))

# training the network
train_num_batches = len(train_data)
train_size = train_set.__len__()

for epoch in range(epochs):
    print("Epoch" + str(epoch))
    for batch, (y, x) in enumerate(train_data):
        batch_size = y.size(dim=0)
        hidden_list = net.backward_imputation(mh_step, impute_lrs, ita, loss_func_sum, sigma_list, x, y)
        for p in net.parameters():
            p.grad = None

        for layer_index in range(num_hidden + 1):
            adj_factor = batch_size/train_size
            hidden_likelihood = -adj_factor * net.likelihood("p", hidden_list, layer_index, loss_func_sum, sigma_list, x, y)

            optimizer = optimizer_list[layer_index]
            hidden_likelihood.backward()
            optimizer.step()

    train_loss, train_correct = 0, 0
    with torch.no_grad():
        for batch, (y, x) in enumerate(train_data):
            pred = net(x)
            train_loss += loss(pred, y).item()
            if regression_flag is False:
                train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= train_num_batches
    train_loss_path[epoch] = train_loss
    print(f"Avg train loss: {train_loss:>8f} \n")
    if regression_flag is False:
        train_correct /= train_size
        train_accuracy_path[epoch] = train_correct
        print(f"train accuracy: {train_correct:>8f} \n")

para = list(net.parameters())
print(para[1])
print(para[3])
print(para[5])
print(para[7])
print(para_sim[1])
print(para_sim[3])
print(para_sim[5])
print(para_sim[7])

np.savetxt(os.path.join(PATH, 'train_loss.txt'), train_loss_path, fmt="%s")
if regression_flag is False:
    np.savetxt(os.path.join(PATH, 'train_accuracy.txt'), train_accuracy_path, fmt="%s")
















