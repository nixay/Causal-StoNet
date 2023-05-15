import torch
import torch.nn as nn
import numpy as np


class StoNet_Causal(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim, treat_layer, treat_node):
        """
        initialize the network
        num_hidden: int
            number of hidden layers
        hidden_dim: list of ints
            dimension of each hidden layer
        input_dim: int
            dimension of network input
        output_dim: int
            dimension of network output
        treat_layer: int
            the layer with treatment variable
        treat_node: list of int
            the hidden node that the treatment variables are located at
        """
        super(StoNet_Causal, self).__init__()
        self.num_hidden = num_hidden
        self.treat_layer = treat_layer
        self.treat_node = treat_node
        self.module_list = []

        self.module_list.append(nn.Linear(input_dim, hidden_dim[0]))
        self.add_module(str(0), self.module_list[0])

        for i in range(self.num_hidden - 1):
            self.module_list.append(nn.Sequential(nn.Tanh(),
                                                  nn.Linear(hidden_dim[i], hidden_dim[i + 1])))
            self.add_module(str(i+1), self.module_list[i+1])

        self.module_list.append(nn.Sequential(nn.Tanh(),
                                              nn.Linear(hidden_dim[-1], output_dim)))
        self.add_module(str(self.num_hidden), self.module_list[self.num_hidden])

        self.prune_flag = 0
        self.mask = None

        self.sse = nn.MSELoss(reduction='sum')

        if isinstance(self.treat_node, (list, tuple, np.ndarray)):
            self.treat_loss = nn.CrossEntropyLoss(reduction='sum')
        else:
            self.treat_loss = nn.BCEWithLogitsLoss(reduction='sum')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, treat, temperature=1):
        if self.prune_flag == 1:
            for name, para in self.named_parameters():
                para.data[self.mask[name]] = 0

        for layer_index in range(self.num_hidden+1):
            x = self.module_list[layer_index](x)
            if layer_index == self.treat_layer:
                logits = torch.clone(x[:, self.treat_node])
                if isinstance(self.treat_node, (list, tuple, np.ndarray)):
                    ps = torch.softmax(logits/temperature, dim=1)
                else:
                    ps = torch.sigmoid(logits/temperature)
                x[:, self.treat_node] = treat
        return x, ps

    def set_prune(self, user_mask):
        self.mask = user_mask
        self.prune_flag = 1

    def cancel_prune(self):
        self.prune_flag = 0
        self.mask = None

    def likelihood(self, forward_hidden, hidden_list, layer_index, outcome_loss, sigma_list, y, temperature=1):
        if layer_index == 0:  # log_likelihood(Y_1|X)
            likelihood = -self.sse(forward_hidden, hidden_list[layer_index]) / (2 * sigma_list[
                layer_index])

        elif layer_index == self.treat_layer:  # log_likelihood(Y_i, A|Y_{i-1})
            z = self.module_list[layer_index](hidden_list[layer_index - 1])

            z_treat = z[:, self.treat_node]
            treat = hidden_list[layer_index][:, self.treat_node]
            likelihood_treat = -self.treat_loss(z_treat/temperature, treat)

            if isinstance(self.treat_node, (list, tuple, np.ndarray)):
                lower = self.treat_node[0]
                upper = self.treat_node[-1]
            else:
                lower = self.treat_node
                upper = self.treat_node

            z_rest_1 = z[:, 0:lower]
            temp1 = hidden_list[layer_index][:, 0:lower]
            likelihood_rest_1 = -self.sse(z_rest_1, temp1)/(2 * sigma_list[layer_index])

            z_rest_2 = z[:, upper + 1:]
            temp2 = hidden_list[layer_index][:, upper + 1:]
            likelihood_rest_2 = -self.sse(z_rest_2, temp2)/(2 * sigma_list[layer_index])

            likelihood = likelihood_treat + likelihood_rest_1 + likelihood_rest_2

        elif layer_index == self.num_hidden:  # log_likelihood(Y|Y_h)
            likelihood = -outcome_loss(self.module_list[layer_index](hidden_list[layer_index - 1]), y) / (
                    2 * sigma_list[self.num_hidden])

        else:  # log_likelihood(Y_i|Y_i-1)
            likelihood = -self.sse(self.module_list[layer_index](hidden_list[layer_index - 1]),
                                   hidden_list[layer_index]) / (2 * sigma_list[layer_index])
        return likelihood

    def backward_imputation(self, mh_step, impute_lrs, alpha, outcome_loss, sigma_list, x, treat, y, temperature=1):
        # initialize momentum term and hidden units
        hidden_list, momentum_list = [], []
        hidden_list.append(self.module_list[0](x).detach())
        momentum_list.append(torch.zeros_like(hidden_list[-1]))
        for layer_index in range(self.num_hidden - 1):
            hidden_list.append(self.module_list[layer_index + 1](hidden_list[-1]).detach())
            momentum_list.append(torch.zeros_like(hidden_list[-1]))
            if layer_index + 1 == self.treat_layer:
                hidden_list[-1][:, self.treat_node] = treat
        for i in range(self.num_hidden):
            hidden_list[i].requires_grad = True
        with torch.no_grad():
            forward_hidden = torch.clone(hidden_list[0])

        # backward imputation by SGHMC
        for step in range(mh_step):
            for layer_index in reversed(range(self.num_hidden)):
                hidden_list[layer_index].grad = None

                hidden_likelihood1 = self.likelihood(forward_hidden, hidden_list, layer_index + 1, outcome_loss, sigma_list,
                                                     y, temperature)
                hidden_likelihood2 = self.likelihood(forward_hidden, hidden_list, layer_index, outcome_loss, sigma_list,
                                                     y, temperature)

                hidden_likelihood1.backward()
                hidden_likelihood2.backward()

                lr = impute_lrs[layer_index]
                with torch.no_grad():
                    momentum_list[layer_index] = (1 - alpha) * momentum_list[layer_index] + lr * hidden_list[
                        layer_index].grad + torch.FloatTensor(hidden_list[layer_index].shape).to(self.device).normal_().mul(
                        np.sqrt(2*alpha))
                    if layer_index == self.treat_layer:
                        # treatment node will not be updated
                        momentum_list[layer_index][:, self.treat_node] = torch.zeros_like(treat)

                    hidden_list[layer_index].data += lr * momentum_list[layer_index]

        return hidden_list
