import torch
import torch.nn as nn
import numpy as np


class StoNet_Causal(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim, treat_layer, treat_node):
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
        self.treat_loss = nn.BCEWithLogitsLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, treat):
        if self.prune_flag == 1:
            for name, para in self.named_parameters():
                para.data[self.mask[name]] = 0

        x = self.module_list[0](x)
        ps = 0  # propensity score
        for layer_index in range(self.num_hidden):
            x = self.module_list[layer_index + 1](x)
            if layer_index + 1 == self.treat_layer:
                score = torch.clone(x[:, self.treat_node])
                ps = score.exp()/(1 + score.exp())
                x[:, self.treat_node] = treat
        return x, ps

    def set_prune(self, user_mask):
        self.mask = user_mask
        self.prune_flag = 1

    def cancel_prune(self):
        self.prune_flag = 0
        self.mask = None

    def likelihood(self, forward_hidden, hidden_list, layer_index, loss_sum, sigma_list, y):
        if layer_index == 0:  # log_likelihood(Y_1|X)
            likelihood = -self.sse(forward_hidden, hidden_list[layer_index]) / (2 * sigma_list[
                layer_index])

        elif layer_index == self.treat_layer:  # log_likelihood(Y_i, A|Y_{i-1})
            z = self.module_list[layer_index](hidden_list[layer_index - 1])

            z_treat = z[:, self.treat_node]
            treat = hidden_list[layer_index][:, self.treat_node]
            likelihood_treat = -self.treat_loss(z_treat, treat)/(2 * sigma_list[layer_index])

            z_rest_1 = z[:, 0:self.treat_node]
            temp1 = hidden_list[layer_index][:, 0:self.treat_node]
            likelihood_rest_1 = -self.sse(z_rest_1, temp1)/(2 * sigma_list[layer_index])

            z_rest_2 = z[:, self.treat_node + 1:]
            temp2 = hidden_list[layer_index][:, self.treat_node + 1:]
            likelihood_rest_2 = -self.sse(z_rest_2, temp2)/(2 * sigma_list[layer_index])

            likelihood = likelihood_treat + likelihood_rest_1 + likelihood_rest_2

        elif layer_index == self.num_hidden:  # log_likelihood(Y|Y_h)
            likelihood = -loss_sum(self.module_list[layer_index](hidden_list[layer_index-1]), y) / (
                    2 * sigma_list[self.num_hidden])

        else:  # log_likelihood(Y_i|Y_i-1)
            likelihood = -self.sse(self.module_list[layer_index](hidden_list[layer_index - 1]),
                                   hidden_list[layer_index]) / (2 * sigma_list[layer_index])
        return likelihood

    def backward_imputation(self, mh_step, impute_lrs, alpha, temperature, loss_sum, sigma_list, x, treat, y):
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

                hidden_likelihood1 = self.likelihood(forward_hidden, hidden_list, layer_index + 1, loss_sum, sigma_list, y)
                hidden_likelihood2 = self.likelihood(forward_hidden, hidden_list, layer_index, loss_sum, sigma_list, y)

                hidden_likelihood1.backward()
                hidden_likelihood2.backward()

                lr = impute_lrs[layer_index]
                with torch.no_grad():
                    momentum_list[layer_index] = (1 - alpha) * momentum_list[layer_index] + lr * hidden_list[
                        layer_index].grad + torch.FloatTensor(hidden_list[layer_index].shape).to(self.device).normal_().mul(
                        np.sqrt(temperature * alpha))
                    if layer_index == self.treat_layer:
                        # treatment node will not be updated
                        momentum_list[layer_index][:, self.treat_node] = torch.zeros_like(treat)

                    hidden_list[layer_index].data += lr * momentum_list[layer_index]

        return hidden_list
