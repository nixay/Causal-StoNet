import torch
import torch.nn as nn
from network_replicate import Net
from data import SimulationData_Cont
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import SGD


# sy's network
class Net2(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim):
        super(Net2, self).__init__()
        self.num_hidden = num_hidden

        self.fc = nn.Linear(input_dim, hidden_dim[0])
        self.fc_list = []

        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.add_module('fc' + str(i + 2), self.fc_list[-1])
        self.fc_list.append(nn.Linear(hidden_dim[-1], output_dim))
        self.add_module('fc' + str(num_hidden + 1), self.fc_list[-1])

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        for i in range(self.num_hidden - 1):
            x = torch.tanh(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x

device = torch.device("cpu")

# training parameters
loss_func_sum = nn.MSELoss(reduction='sum')
sse = nn.MSELoss(reduction='sum')
loss = nn.MSELoss()
impute_lrs = [1e-7, 1e-7, 1e-7]
ita = 0.5
batch_size = 5
para_lrs = [1e-4]
para_momentum = 0.9
para_lrs = [1e-4, 1e-5, 1e-6, 1e-7]

# simulation dataset
train_data = DataLoader(SimulationData_Cont(1, 500), batch_size=batch_size)
test_data = DataLoader(SimulationData_Cont(2, 500), batch_size=batch_size)
dim_data = len(train_data.dataset)

# get input size
train_iter = iter(train_data)
y, treat, x = next(train_iter)
input_dim = x[0].size(dim=0)

# network parameters
num_hidden = 3
hidden_dim = [4, 3, 2]
output_dim = 1
sigma_list = [1e-3, 1e-4, 1e-5, 1e-6]
mh_step = 1

# network
net = Net(num_hidden, hidden_dim, input_dim, output_dim)
net2 = Net2(num_hidden, hidden_dim, input_dim, output_dim)

para = list(net.parameters())
net2.state_dict()['fc.bias'][:] = para[1]
net2.state_dict()['fc2.bias'][:] = para[3]
net2.state_dict()['fc3.bias'][:] = para[5]
net2.state_dict()['fc4.bias'][:] = para[7]
net2.state_dict()['fc.weight'][:] = para[0]
net2.state_dict()['fc2.weight'][:] = para[2]
net2.state_dict()['fc3.weight'][:] = para[4]
net2.state_dict()['fc4.weight'][:] = para[6]

forward_hidden = net2.fc(x)


alpha = 1
temperature = [0, 0, 0]

# define optimizer
optimizer_list = []
for i in range(num_hidden+1):
    optimizer_list.append(SGD(net.module_dict[str(i)].parameters(), lr=para_lrs[i], momentum=para_momentum))

optimizer_list2 = []
optimizer_list2.append(torch.optim.SGD(net2.fc.parameters(), lr=para_lrs[0], momentum=para_momentum))
for i in range(num_hidden):
    optimizer_list2.append(torch.optim.SGD(net2.fc_list[i].parameters(), lr=para_lrs[i+1], momentum=para_momentum))
#######################################################################################################################
print('Test 1')

hidden_list = net.backward_imputation(mh_step, impute_lrs, ita, loss_func_sum, sigma_list, x, y)
print('hidden list', hidden_list)
for p in net.parameters():
    if p.grad is not None:
        p.grad.zero_()

for layer_index in range(num_hidden + 1):
    adj_factor = 1/batch_size
    hidden_likelihood = -adj_factor * net.likelihood("p", hidden_list, layer_index, loss_func_sum, sigma_list, x, y)
    print(str(layer_index) + 'hidden_likelihood', hidden_likelihood)

    optimizer = optimizer_list[layer_index]
    optimizer.zero_grad()
    hidden_likelihood.backward()
    for p in net.parameters():
        print('parameter grad', p.grad)
    optimizer.step()

pred = net.forward(x)
train_loss = loss(pred, y).item()
print(f"Avg train loss: {train_loss:>8f} \n")

#######################################################################################################################
print('Test 2')

# update parameter for first layer
loss = sse(net2.fc(x), hidden_list[0]) / (2 * sigma_list[0]) / batch_size
print(str(0) + 'loss', loss)

for para in net2.fc.parameters():
    loss = loss

optimizer_list[0].zero_grad()
loss.backward()
for p in net2.parameters():
    print('parameter grad', p.grad)
optimizer_list[0].step()


for layer_index in range(num_hidden):
    # update parameters layer by layer
    if layer_index == num_hidden - 1:
        loss = loss_func_sum(net2.fc_list[layer_index](torch.tanh(hidden_list[layer_index])),
                             y) / (2 * sigma_list[layer_index + 1]) / batch_size
        print(str(layer_index) + 'loss', loss)
    else:
        loss = sse(net2.fc_list[layer_index](torch.tanh(hidden_list[layer_index])),
                   hidden_list[layer_index + 1]) / (2 * sigma_list[layer_index + 1]) / batch_size
        print(str(layer_index) + 'loss', loss)

    for para in net2.fc_list[layer_index].parameters():
        loss = loss

    optimizer_list[layer_index + 1].zero_grad()
    loss.backward()
    for p in net2.parameters():
        print('parameter grad', p.grad)
    optimizer_list[layer_index + 1].step()
