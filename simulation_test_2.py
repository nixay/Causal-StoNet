import torch
import torch.nn as nn
from network import Net
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import SGD
import numpy as np
import os
import errno
import json
import argparse

parser = argparse.ArgumentParser(description='Simulation of Causal StoNet with sparsity')
parser.add_argument('--data_seed', default=1, type=int, help='set seed')
parser.add_argument('--partition_seed', default=2, type=int, help='set seed')
parser.add_argument('--regression', dest='regression_flag', action='store_true', help='true for regression')
parser.add_argument('--classification', dest='regression_flag', action='store_false', help='false for classification')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for DataLoader')
parser.add_argument('--train_size', default=10000, type=int, help='size of training set')
parser.add_argument('--val_size', default=2000, type=int, help='size of validation set')
parser.add_argument('--train_epoch', default=1000, type=int, help='number of training epochs')
parser.add_argument('--fine_tune_epoch', default=200, type=int, help='number of finetuning epochs')
parser.add_argument('--num_seed', default=5, type=int, help='number of runs for each pruning processs')
parser.add_argument('--batch_size', default=50, type=int, help='batch size')

args = parser.parse_args()
########################################################################################################################
# set seed for generating dataset
data_seed = args.data_seed
partition_seed = args.partition_seed

# set number of runs for the network
num_seed = args.num_seed

# task
regression_flag = args.regression_flag

# number of workers for DataLoader
num_workers = args.num_workers

# StoNet for generating data
num_hidden = 3
treat_layer = 1
hidden_dim = [4, 3, 2]
treat_node = 1
sim_input_dim = 5

if regression_flag:
    output_dim = 1
else:
    output_dim = 2

net_sim = Net(num_hidden, hidden_dim, sim_input_dim, output_dim, treat_layer, treat_node)
net_sim.state_dict()['0.weight'][:] = torch.tensor([[2, -1, 1, 0, 0],
                                                    [0, -2, 0, 1, 0],
                                                    [1, 0, 0, 2, -2],
                                                    [-1, 1, 2, -1, 1]])
net_sim.state_dict()['0.bias'][:] = torch.zeros_like(net_sim.state_dict()['0.bias'])
net_sim.state_dict()['1.1.weight'][:] = torch.tensor([[0, 2, -1, 0],
                                                      [1, 0, -2, -1],
                                                      [-1, 1, 2, 0]])
net_sim.state_dict()['1.1.bias'][:] = torch.zeros_like(net_sim.state_dict()['1.1.bias'])
net_sim.state_dict()['2.1.weight'][:] = torch.tensor([[0, 2, 1],
                                                      [1, -1, 0]])
net_sim.state_dict()['2.1.bias'][:] = torch.zeros_like(net_sim.state_dict()['2.1.bias'])
net_sim.state_dict()['3.1.weight'][:] = torch.tensor([[1, 2]])
net_sim.state_dict()['3.1.bias'][:] = torch.zeros_like(net_sim.state_dict()['3.1.bias'])


class SimStoNet_Cont(Dataset):
    """
    generate simulation data using StoNet for regression task

    seed: float
        random seed
    true_input_size: int
        the number of input variables that are used to generate y
    total_input_size: int
        the total number of input variables
    data_size: int
        the size of the generated dataset
    """
    def __init__(self, seed, true_input_size, total_input_size, data_size):
        self.data_size = data_size
        self.treat = np.zeros(self.data_size)
        self.x = np.zeros([self.data_size] + [total_input_size])

        np.random.seed(seed)
        torch.manual_seed(seed)
        for i in range(self.data_size):
            treat_temp = np.random.binomial(1, 0.5)

            ee = np.random.normal(0, 1)
            x_temp = np.repeat(ee, true_input_size)
            for j in range(true_input_size):
                x_temp[j] += np.random.normal(0, 1)
            x_temp /= np.sqrt(2)
            x_temp = np.concatenate((x_temp, np.zeros(total_input_size - true_input_size)))

            self.treat[i] = treat_temp
            self.x[i:] = x_temp

        self.treat = torch.FloatTensor(self.treat)
        self.x = torch.FloatTensor(self.x)

        self.y = net_sim.forward(self.x[:, :true_input_size], self.treat).data

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        return y, treat, x


class SimStoNet_Bin(Dataset):
    """
    generate simulation data using StoNet for binary classification task

    seed: float
        random seed
    true_input_size: int
        the number of input variables that are used to generate y
    total_input_size: int
        the total number of input variables
    data_size: int
        the size of the generated dataset
    """
    def __init__(self, seed, true_input_size, total_input_size, data_size):
        self.data_size = data_size
        self.treat = np.zeros(self.data_size)
        self.x = np.zeros([self.data_size] + [total_input_size])

        np.random.seed(seed)
        torch.manual_seed(seed)
        for i in range(int(self.data_size)):
            treat_temp = np.random.binomial(1, 0.5)

            ee = np.random.normal(0, 1)
            x_temp = np.repeat(ee, true_input_size)
            for j in range(true_input_size):
                x_temp[j] += np.random.normal(0, 1)
            x_temp /= np.sqrt(2)
            x_temp = np.concatenate((x_temp, np.zeros(total_input_size - true_input_size)))

            self.treat[i] = treat_temp
            self.x[i:] = x_temp

        self.treat = torch.FloatTensor(self.treat)
        self.x = torch.FloatTensor(self.x)

        score = net_sim.forward(self.x[:, :true_input_size], self.treat).data
        label_temp = score.argmax(1)
        self.label = torch.LongTensor(label_temp)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        label = self.label[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        return label, treat, x


########################################################################################################################
# network setup
num_hidden = 3
treat_depth = 1
hidden_dim = [6, 4, 3]
treat_node = 1

if regression_flag:
    output_dim = 1
else:
    output_dim = 2

# load data for training
batch_size = args.batch_size
train_size = args.train_size
val_size = args.val_size
data_size = train_size + val_size
if regression_flag:
    data = SimStoNet_Cont(data_seed, sim_input_dim, 1000, data_size)
    train_set, val_set = random_split(data, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(partition_seed))
else:
    data = SimStoNet_Bin(data_seed, sim_input_dim, 1000, data_size)
    train_set, val_set = random_split(data, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(partition_seed))

train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

_, _, x_temp = next(iter(train_data))
input_dim = x_temp[0].size(dim=0)
train_size = train_set.__len__()
val_size = val_set.__len__()

# imputation parameters
impute_lrs = [1e-7, 1e-7, 1e-7]
ita = 0.5
mh_step = 1
sigma_list = [1e-6, 1e-5, 1e-4, 1e-3]

# training parameters
para_momentum = 0.9
para_lrs = [1e-4, 1e-5, 1e-6, 1e-7]
training_epochs = args.train_epoch

# sparsity parameters
fine_tune_epoch = args.fine_tune_epoch
prior_sigma_0 = 0.0005
prior_sigma_1 = 0.01
lambda_n = 0.00001

# parameter prior
c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1

# threshold for sparsity
threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
        0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

# settings for loss functions
if regression_flag:
    loss = nn.MSELoss()
    loss_sum = nn.MSELoss(reduction='sum')
else:
    loss = nn.CrossEntropyLoss()
    loss_sum = nn.CrossEntropyLoss(reduction='sum')

# path to save the result
if regression_flag:
    base_path = os.path.join('.', 'stonet_sparsity', 'result', 'sim', 'regression')
else:
    base_path = os.path.join('.', 'stonet_sparsity', 'result', 'sim', 'classification')
spec = str(impute_lrs) + '_' + str(para_lrs) + '_' + str(hidden_dim) + '_' + str(training_epochs) + '_' + \
       str(fine_tune_epoch)
base_path = os.path.join(base_path, spec)


def optimization(net, epochs):
    """
    train the network
    net: Net object
        the network to be trained
    epochs: float
        the number of epochs that the network is trained for
    """
    # save parameter values, indicator variables, and selected input
    para_path = {}
    para_gamma_path = {}
    var_gamma = {}
    for name, para in net.named_parameters():
        para_path[name] = {}
        para_gamma_path[name] = {}
        for epoch in range(epochs):
            para_path[name][str(epoch)] = np.zeros(para.shape)
            para_path[name][str(epoch)] = np.zeros(para.shape)

    # save training and validation loss
    performance = {}
    train_loss_path = []
    val_loss_path = []
    performance['train_loss'] = train_loss_path
    performance['val_loss'] = val_loss_path
    if regression_flag is False:
        train_accuracy_path = []
        val_accuracy_path = []
        performance['train_acc'] = train_accuracy_path
        performance['val_acc'] = val_accuracy_path

    # define optimizer for parameter update
    optimizer_list = []
    for j in range(num_hidden + 1):
        optimizer_list.append(SGD(net.module_dict[str(j)].parameters(), lr=para_lrs[j], momentum=para_momentum))

    # training the network
    train_num_batches = len(train_data)
    val_num_batches = len(val_data)

    # save hidden likelihood for to calculate BIC
    hidden_likelihood = np.zeros(num_hidden+1)

    for epoch in range(epochs):
        print("Epoch" + str(epoch))
        # train loop
        for y, treat, x in train_data:
            # backward imputation
            hidden_list = net.backward_imputation(mh_step, impute_lrs, ita, loss_sum, sigma_list, x, y, treat)

            # parameter update
            for para in net.parameters():
                para.grad = None

            with torch.no_grad():  # prior gradient
                for para in net.parameters():
                    temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                    prior_grad = temp.div(train_size)
                    para.grad = -prior_grad

            adj_factor = batch_size / train_size
            for layer_index in range(num_hidden + 1):  # data likelihood gradient. update layer by layer
                likelihood = net.likelihood("p", hidden_list, layer_index, loss_sum, sigma_list, x, y)
                hidden_likelihood[layer_index] = likelihood
                likelihood = -adj_factor * likelihood
                optimizer = optimizer_list[layer_index]
                likelihood.backward()
                optimizer.step()

        train_loss, train_correct = 0, 0
        with torch.no_grad():
            for y, treat, x in train_data:
                pred = net.forward(x, treat)
                train_loss += loss(pred, y).item()
                if regression_flag is False:
                    train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        train_loss /= train_num_batches
        train_loss_path.append(train_loss)
        print(f"Avg train loss: {train_loss:>8f} \n")

        if regression_flag is False:
            train_correct /= train_size
            train_accuracy_path.append(train_correct)
            print(f"train accuracy: {train_correct:>8f} \n")

        # validation loop
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for y, treat, x in val_data:
                pred = net.forward(x, treat)
                val_loss += loss(pred, y).item()
                if regression_flag is False:
                    val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        val_loss /= val_num_batches
        val_loss_path.append(val_loss)
        print(f"Avg val loss: {val_loss:>8f} \n")

        if regression_flag is False:
            val_correct /= val_size
            val_accuracy_path.append(val_correct)
            print(f"val accuracy: {val_correct:>8f} \n")

        # save parameter values and select connections
        for name, para in net.named_parameters():
            para_path[name][str(epoch)] = para.data.numpy()
            para_gamma_path[name][str(epoch)] = (para.abs() > threshold).data.numpy()

        # select input variable
        var_ind = para_gamma_path['0.weight'][str(epoch)]
        for i, (name, para) in enumerate(net.named_parameters()):
            if i % 2 == 0 and i > 0:
                var_ind = np.matmul(para_gamma_path[name][str(epoch)], var_ind)
        variable_selected = np.max(var_ind, 0)
        num_selected = np.sum(variable_selected)
        var_gamma[str(epoch)] = variable_selected.tolist()
        print('number of selected input variable:', num_selected)

    return para_path, para_gamma_path, var_gamma, performance, hidden_likelihood


def main():
    dim_list = np.zeros([num_seed])  # total number of non-zero element of the pruned network
    BIC_list = np.zeros([num_seed])
    num_selection_list = np.zeros([num_seed])  # number of selected input variables
    train_loss_list = np.zeros([num_seed])
    val_loss_list = np.zeros([num_seed])
    if regression_flag is False:
        train_accuracy_list = np.zeros([num_seed])
        val_accuracy_list = np.zeros([num_seed])

    for prune_seed in range(num_seed):
        print('number of runs', prune_seed)
        # path to save the result
        PATH = os.path.join(base_path, str(prune_seed))
        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        # training the network
        np.random.seed(prune_seed)
        torch.manual_seed(prune_seed)
        net = Net(num_hidden, hidden_dim, input_dim, output_dim, treat_depth, treat_node)

        para_train, para_gamma_train, var_gamma_train, performance_train, _ = optimization(net, training_epochs)

        # prune network parameters
        with torch.no_grad():
            for name, para in net.named_parameters():
                para.data = torch.FloatTensor(para_train[name][str(training_epochs-1)])

        user_mask = {}
        for name, para in net.named_parameters():
            user_mask[name] = para.abs() < threshold
        net.set_prune(user_mask)

        # save model training results
        num_selection_list[prune_seed] = np.sum(var_gamma_train[str(training_epochs-1)])

        temp_str = [str(int(x)) for x in var_gamma_train[str(training_epochs-1)]]
        temp_str = ' '.join(temp_str)
        filename = PATH + 'selected_variable.txt'
        f = open(filename, 'w')
        f.write(temp_str)
        f.close()

        for name, para in net.named_parameters():
            for epoch in range(training_epochs):
                para_gamma_train[name][str(epoch)] = para_gamma_train[name][str(epoch)].tolist()

        para_gamma_file = open(os.path.join(PATH, 'net_para_gamma_train.json'), "w")
        json.dump(para_gamma_train, para_gamma_file, indent="")
        para_gamma_file.close()

        performance_file = open(os.path.join(PATH, 'performance_train.json'), "w")
        json.dump(performance_train, performance_file, indent="")
        performance_file.close()

        performance_file = open(os.path.join(PATH, 'variable_selected_train.json'), "w")
        json.dump(var_gamma_train, performance_file, indent="")
        performance_file.close()

        # refine non-zero network parameters
        para_fine_tune, para_gamma_fine_tune, var_gamma_fine_tune, performance_fine_tune, likelihoods \
            = optimization(net, fine_tune_epoch)

        # save fine tuning results
        for name, para in net.named_parameters():
            for epoch in range(fine_tune_epoch):
                para_gamma_fine_tune[name][str(epoch)] = para_gamma_fine_tune[name][str(epoch)].tolist()

        para_gamma_file = open(os.path.join(PATH, 'net_para_gamma_fine_tune.json'), "w")
        json.dump(para_gamma_fine_tune, para_gamma_file, indent="")
        para_gamma_file.close()

        performance_file = open(os.path.join(PATH, 'performance_fine_tune.json'), "w")
        json.dump(performance_fine_tune, performance_file, indent="")
        performance_file.close()

        performance_file = open(os.path.join(PATH, 'var_gamma_fine_tune.json'), "w")
        json.dump(var_gamma_fine_tune, performance_file, indent="")
        performance_file.close()

        # save training results for this run
        train_loss = performance_fine_tune['train_loss'][-1]
        train_loss_list[prune_seed] = train_loss
        val_loss = performance_fine_tune['val_loss'][-1]
        val_loss_list[prune_seed] = val_loss
        if regression_flag is False:
            train_accuracy = performance_fine_tune['train_acc'][-1]
            train_accuracy_list[prune_seed] = train_accuracy
            val_accuracy = performance_fine_tune['val_acc'][-1]
            val_accuracy_list[prune_seed] = val_accuracy

        # calculate BIC
        with torch.no_grad():
            num_non_zero_element = 0
            for name, para in net.named_parameters():
                num_non_zero_element = num_non_zero_element + para.numel() - net.mask[name].sum()
            dim_list[prune_seed] = num_non_zero_element

            BIC = (np.log(train_size) * num_non_zero_element - 2 * np.sum(likelihoods)).item()
            BIC_list[prune_seed] = BIC

            print("number of non-zero connections:", num_non_zero_element)
            print('BIC:', BIC)

        torch.save(net.state_dict(), os.path.join(PATH, 'model' + str(prune_seed)+'.pt'))

    np.savetxt(os.path.join(base_path, 'Overall_train_loss.txt'), train_loss_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_val_loss.txt'), val_loss_list, fmt="%s")
    if regression_flag is False:
        np.savetxt(os.path.join(base_path, 'Overall_train_accuracy.txt'), train_accuracy_list, fmt="%s")
        np.savetxt(os.path.join(base_path, 'Overall_val_accuracy.txt'), val_accuracy_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_BIC.txt'), BIC_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_non_zero_connections.txt'), dim_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_selected_variables.txt'), num_selection_list, fmt="%s")


if __name__ == '__main__':
    main()
