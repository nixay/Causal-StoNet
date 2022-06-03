import torch
import torch.nn as nn
from network_replicate import Net
from data import SimulationData_Cont
from torch.utils.data import DataLoader
import numpy as np


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
impute_lrs = [1e-7, 1e-7, 1e-7]
ita = 0.5
batch_size = 5
para_lrs = [1e-4]
para_momentum = 0.9

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
#######################################################################################################################
print('Test 1')
# initialize momentum term and hidden units
hidden_list, momentum_list = [], []
hidden_list.append(net.module_dict[str(0)](x).detach())
momentum_list.append(torch.zeros_like(hidden_list[-1]))
for layer_index in range(net.num_hidden - 1):
    hidden_list.append(net.module_dict[str(layer_index + 1)](hidden_list[-1]).detach())
    momentum_list.append(torch.zeros_like(hidden_list[-1]))
print('hidden list', hidden_list)

# backward imputation by SGHMC
for step in range(mh_step):
    print('mh_step', step)
    # update momentum variable and latent varibales
    for layer_index in reversed(range(net.num_hidden)):
        if hidden_list[layer_index].grad is not None:  # clear the gradient of latent variables
            hidden_list[layer_index].grad.zero_()

        hidden_likelihood1 = net.likelihood("l", hidden_list, layer_index+1, loss_func_sum, sigma_list, x, y)
        print('hidden_likelihood1', hidden_likelihood1)
        hidden_likelihood2 = net.likelihood("l", hidden_list, layer_index, loss_func_sum, sigma_list, x, y)
        print('hidden_likelihood2', hidden_likelihood2)
        hidden_likelihood = hidden_likelihood1 + hidden_likelihood2

        print(str(layer_index) + ' hidden likelihood', hidden_likelihood)
        hidden_likelihood.backward()
        print(str(layer_index) + " hidden list grad", hidden_list[layer_index].grad)

        step_proposal_lr = impute_lrs[layer_index]

        with torch.no_grad():
            momentum_list[layer_index] = (1 - alpha) * momentum_list[layer_index] + step_proposal_lr * \
                                         hidden_list[
                                             layer_index].grad + torch.FloatTensor(
                hidden_list[layer_index].shape).to(device).normal_().mul(
                np.sqrt(alpha * step_proposal_lr * temperature[layer_index]))
            # print(hidden_list[layer_index].grad)
            hidden_list[layer_index].data += momentum_list[layer_index]
        print(hidden_list)

#######################################################################################################################
print('Test 2')
# Initialize hidden units by forward pass
hidden_list = []
momentum_list = []
with torch.no_grad():
    hidden_list.append(net2.fc(x))
    momentum_list.append(torch.zeros_like(hidden_list[-1]))
    for i in range(num_hidden - 1):
        hidden_list.append(net2.fc_list[i](torch.tanh(hidden_list[-1])))
        momentum_list.append(torch.zeros_like(hidden_list[-1]))
for i in range(hidden_list.__len__()):
    hidden_list[i].requires_grad = True
with torch.no_grad():
    forward_hidden = torch.clone(hidden_list[0])
print('hidden list', hidden_list)

for repeat in range(mh_step):
    print('mh_step', repeat)
    for layer_index in reversed(range(num_hidden)):
        if hidden_list[layer_index].grad is not None:
            hidden_list[layer_index].grad.zero_()
        if layer_index == num_hidden - 1:
            hidden_likelihood = -loss_func_sum(net2.fc_list[layer_index](torch.tanh(hidden_list[layer_index])),
                                               y) / (2 * sigma_list[layer_index + 1])
            print(str(layer_index) + ' hidden likelihood', hidden_likelihood)
        else:
            hidden_likelihood = -sse(net2.fc_list[layer_index](torch.tanh(hidden_list[layer_index])),
                                     hidden_list[layer_index + 1]) / (2 * sigma_list[layer_index + 1])
            print(str(layer_index) + ' hidden likelihood', hidden_likelihood)
        if layer_index == 0:
            hidden_likelihood = hidden_likelihood - sse(forward_hidden, hidden_list[layer_index]) / (2*sigma_list[layer_index])
            print(str(layer_index) + ' hidden likelihood', hidden_likelihood)
        else:
            hidden_likelihood = hidden_likelihood - sse(
                net2.fc_list[layer_index - 1](torch.tanh(hidden_list[layer_index - 1])),
                hidden_list[layer_index]) / (2*sigma_list[layer_index])
            print(str(layer_index) + ' hidden likelihood', hidden_likelihood)

        hidden_likelihood.backward()
        print(str(layer_index) + " hidden list grad", hidden_list[layer_index].grad)
        step_proposal_lr = impute_lrs[layer_index]

        with torch.no_grad():
            momentum_list[layer_index] = (1 - alpha) * momentum_list[layer_index] + step_proposal_lr * \
                                         hidden_list[
                                             layer_index].grad + torch.FloatTensor(
                hidden_list[layer_index].shape).to(device).normal_().mul(
                np.sqrt(alpha * step_proposal_lr * temperature[layer_index]))
            # print(hidden_list[layer_index].grad)
            hidden_list[layer_index].data += momentum_list[layer_index]
        print(hidden_list)

