import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    """
    num_hidden: int
                number of hidden layers
    hidden_dim: list
                dimension of each hidden layer
    input_dim: int
                dimension of network input
    output_dim: int
                dimension of network output
    treat_layer: int
                index of the layer that has treatment variable
    """

    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim, treat_layer, treat_node):
        super(Net, self).__init__()
        self.num_hidden = num_hidden
        self.treat_layer = treat_layer
        self.treat_node = treat_node
        self.module_dict = {}

        self.module_dict[str(0)] = nn.Linear(input_dim, hidden_dim[0])
        self.add_module(str(0), self.module_dict[str(0)])

        for i in range(num_hidden - 1):
            self.module_dict[str(i + 1)] = nn.Sequential(nn.Tanh(),
                                                         nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.add_module(str(i + 1), self.module_dict[str(i + 1)])

        self.module_dict[str(num_hidden)] = nn.Sequential(nn.Tanh(),
                                                          nn.Linear(hidden_dim[-1], output_dim))
        self.add_module(str(num_hidden), self.module_dict[str(num_hidden)])

    def forward(self, x):
        # needed for calculating train loss and test loss
        x = self.module_dict[str(0)](x)
        for layer_index in range(self.num_hidden):
            x = self.module_dict[str(layer_index + 1)](x)
        return x

    def likelihood(self, mode, hidden_list, layer_index, loss_sum, sigma_list, x, y, treat):
        """
        Calculate log-likelihood P(Y_i|Y_i-1) for hidden layers

        mode: str
            scenario of calculating gradient
            "l": calculate gradient for latent variables
            "p": calculate gradient for network parameters
        hidden_list: list of lists
            hidden variables in each layer (i.e. Y_i's)
        layer_index: int
            index of each layer, starting from 0
        loss_sum:
            functions to calculate log_likelihood for the last hidden layer
        sigma_list: list
            variance of the variables for each layer. dim = num_hidden + 1
        x: tensor
            input of the network.
        y: tensor
            output of the network.
        treat: int
            treatment variable, binary
        """
        if mode == "l":  # latent sampling
            for i in range(self.num_hidden):
                hidden_list[i].requires_grad = True
            for p in self.parameters():
                p.requires_grad = False
        if mode == "p":  # parameter update
            for i in range(self.num_hidden):
                hidden_list[i].requires_grad = False
            for p in self.parameters():
                p.requires_grad = True
        sse = nn.MSELoss(reduction='sum')
        treat_loss_sum = nn.BCEWithLogitsLoss(reduction='sum')

        if layer_index == 0:  # log_likelihood(Y_1|X)
            likelihood = -sse(self.module_dict[str(layer_index)](x), hidden_list[layer_index]) / (2 * sigma_list[
                layer_index])

        elif layer_index == self.treat_layer:  # log_likelihood(Y_i, A|Y_{i-1})
            z = self.module_dict[str(layer_index)](hidden_list[layer_index - 1])
            z1 = z[:, self.treat_node]
            z_rest = torch.cat((z[:, 0:self.treat_node], z[:, self.treat_node + 1:]), 1)
            temp1 = hidden_list[self.treat_layer][:, 0:self.treat_node]
            temp2 = hidden_list[self.treat_layer][:, self.treat_node + 1:]
            treat_layer_rest = torch.cat((temp1, temp2), 1)
            likelihood = -treat_loss_sum(z1, treat) - sse(z_rest, treat_layer_rest) / (2 * sigma_list[layer_index])

        elif layer_index == self.num_hidden:  # log_likelihood(Y|Y_h)
            likelihood = -loss_sum(self.module_dict[str(self.num_hidden)](hidden_list[-1]), y) / (
                    2 * sigma_list[self.num_hidden])

        else:  # log_likelihood(Y_i|Y_i-1)
            likelihood = -sse(self.module_dict[str(layer_index)](hidden_list[layer_index - 1]),
                              hidden_list[layer_index]) / (2 * sigma_list[layer_index])
        return likelihood

    def backward_imputation(self, mh_step, lrs, ita, loss_sum, sigma_list, x, y, treat):
        """
        backward imputation using SGHMC.

        mh_steps: int
            Monte Carlo step number
        lrs: list
            learning rate for SGHMC. dim = num_hidden
        ita: float
            friction coefficient for SGHMC.
        loss_sum:
            functions to calculate log_likelihood for the last hidden layer
        sigma_list: list
            variance of the variables for each layer. dim = num_hidden + 1
        x: tensor
            input of the network. dim = input_dim
        y: tensor
            output of the network
        treat: int
            treatment variable, binary
        """
        # initialize momentum term and hidden units
        hidden_list, momentum_list = [], []
        hidden_list.append(self.module_dict[str(0)](x).detach())
        momentum_list.append(torch.zeros_like(hidden_list[-1]))
        for layer_index in range(self.num_hidden - 1):
            hidden_list.append(self.module_dict[str(layer_index + 1)](hidden_list[-1]).detach())
            momentum_list.append(torch.zeros_like(hidden_list[-1]))
        # assign treatment to treatment node
        hidden_list[self.treat_layer][:, self.treat_node] = treat

        # backward imputation by SGHMC
        for step in range(mh_step):
            for layer_index in reversed(range(self.num_hidden)):
                hidden_list[layer_index].grad = None

                hidden_likelihood1 = self.likelihood("l", hidden_list, layer_index + 1, loss_sum, sigma_list, x, y, treat)
                hidden_likelihood2 = self.likelihood("l", hidden_list, layer_index, loss_sum, sigma_list, x, y, treat)

                hidden_likelihood1.backward()
                hidden_likelihood2.backward()

                alpha = lrs[layer_index] * ita
                temperature = np.random.normal(0, 1, momentum_list[layer_index].size())
                lr = lrs[layer_index]
                with torch.no_grad():
                    momentum_list[layer_index] = (1 - alpha) * momentum_list[layer_index] + lr * hidden_list[
                        layer_index].grad + np.sqrt(2 * alpha) * temperature
                    if layer_index == self.treat_layer:
                        # treatment node will not be updated
                        momentum_list[layer_index][:, self.treat_node] = torch.zeros_like(treat)
                    hidden_list[layer_index].data += momentum_list[layer_index]

        # turn off the gradient tracking after final update of hidden_list
        for layer_index in range(self.num_hidden):
            hidden_list[layer_index].requires_grad = False
        return hidden_list
