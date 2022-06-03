import torch
import torch.nn as nn
from network_replicate import Net
from data import SimulationData_Cont
from torch.utils.data import DataLoader
import numpy as np

# training parameters
loss_func_sum = nn.MSELoss(reduction='sum')
sse = nn.MSELoss(reduction='sum')
impute_lrs = [1e-7, 1e-7, 1e-7]
ita = 0.5
batch_size = 5
para_lrs = [1e-4]
para_momentum = 0.9

# simulation dataset
train_data = DataLoader(SimulationData_Cont(1, 500), batch_size=batch_size)
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

net = Net(num_hidden, hidden_dim, input_dim, output_dim)
hidden_list = net.backward_imputation(mh_step, impute_lrs, ita, loss_func_sum, sigma_list, x, y)

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

#######################################################################################################################
print("Test 1")
# method used by me
for layer_index in range(num_hidden):
    if hidden_list[layer_index].grad is not None:
        hidden_list[layer_index].grad.zero_()
    hidden_likelihood = net.likelihood("l", hidden_list, layer_index, loss_func_sum, sigma_list, x, y)
    print(hidden_likelihood)
    hidden_likelihood.backward()

for layer_index in range(num_hidden):
    print(hidden_list[layer_index].grad)

#######################################################################################################################
print("Test 2")
# calculate likelihood separately and then differentiate separately
for layer_index in range(num_hidden):
    if hidden_list[layer_index].grad is not None:  # clear the gradient of latent variables
        hidden_list[layer_index].grad.zero_()

l0 = -sse(net.module_dict[str(0)](x), hidden_list[0]) / (2 * sigma_list[0])
l1 = -sse(net.module_dict[str(1)](hidden_list[0]), hidden_list[1]) / (2 * sigma_list[1])
l2 = -sse(net.module_dict[str(2)](hidden_list[1]), hidden_list[2]) / (2 * sigma_list[2])
l3 = -sse(net.module_dict[str(3)](hidden_list[2]), y) / (2 * sigma_list[3])

l0.backward()
# print(hidden_list[0].grad)
l1.backward()
# print(hidden_list[0].grad)
# print(hidden_list[1].grad)
l2.backward()
# print(hidden_list[1].grad)
# print(hidden_list[2].grad)
l3.backward()
# print(hidden_list[2].grad)

print(l0)
print(l1)
print(l2)
print(l3)

for layer_index in range(num_hidden):
    print(hidden_list[layer_index].grad)

#######################################################################################################################
print("Test 3")
net2 = Net2(num_hidden, hidden_dim, input_dim, 1)

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

for layer_index in reversed(range(num_hidden)):
    if hidden_list[layer_index].grad is not None:
        hidden_list[layer_index].grad.zero_()
    if layer_index == num_hidden - 1:
        hidden_likelihood = -loss_func_sum(net2.fc_list[layer_index](torch.tanh(hidden_list[layer_index])),
                                           y) / (2*sigma_list[layer_index + 1])
        print('last layer hidden likelihood', hidden_likelihood)
    else:
        hidden_likelihood = -sse(net2.fc_list[layer_index](torch.tanh(hidden_list[layer_index])),
                                 hidden_list[layer_index + 1]) / (2*sigma_list[layer_index + 1])
        print(str(layer_index) + ' hidden likelihood', hidden_likelihood)
    if layer_index == 0:
        hidden_likelihood = hidden_likelihood - sse(forward_hidden, hidden_list[layer_index]) / (2*sigma_list[layer_index])
        print('first layer hidden likelihood', - sse(forward_hidden, hidden_list[layer_index]) / (2*sigma_list[layer_index]))
    else:
        hidden_likelihood = hidden_likelihood - sse(
            net2.fc_list[layer_index - 1](torch.tanh(hidden_list[layer_index - 1])),
            hidden_list[layer_index]) / (2*sigma_list[layer_index])
    hidden_likelihood.backward()
    print(hidden_likelihood)

for layer_index in range(num_hidden):
    print(hidden_list[layer_index].grad)

#######################################################################################################################





