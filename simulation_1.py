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
from scipy.stats import truncnorm
import time

parser = argparse.ArgumentParser(description='Simulation of Variable Selection for Causal StoNet')
# Basic Setting
# simulation setting
parser.add_argument('--data_seed', default=1, type=int, help='set seed for data generation')
parser.add_argument('--partition_seed', default=1, type=int, help='set seed for dataset partition')

# dataset setting
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for DataLoader')
parser.add_argument('--train_size', default=10000, type=int, help='size of training set')
parser.add_argument('--val_size', default=1000, type=int, help='size of validation set')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# Parameter for StoNet
# model
parser.add_argument('--layer', default=3, type=int, help='number of hidden layer (not including the treatment layer')
parser.add_argument('--unit', default=[6, 4, 3], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[1e-6, 1e-5, 1e-4, 1e-3], type=float, nargs='+',
                    help='variance of each layer for the model')
parser.add_argument('--depth', default=1, type=int, help='number of layers before the treatment layer')
parser.add_argument('--treat_node', default=1, type=int, help='the position of the treatment variable')

# training setting
parser.add_argument('--train_epoch', default=1000, type=int, help='total number of training epochs')
parser.add_argument('--mh_step', default=1, type=int, help='number of SGHMC step for imputation')
parser.add_argument('--impute_lr', default=[1e-7, 1e-7, 1e-7, 1e-7], type=float, nargs='+', help='step size in imputation')
parser.add_argument('--ita', default=0.5, type=float, help='friction coefficient for SGHMC')
parser.add_argument('--para_lr_train', default=[1e-5, 1e-6, 1e-7, 1e-8], type=float, nargs='+',
                    help='step size of parameter update for training stage')
parser.add_argument('--para_momentum', default=0.9, type=float, help='momentum parameter for parameter update')

# Parameters for Sparsity
parser.add_argument('--num_run', default=1, type=int, help='Number of different initialization used to train the model')
parser.add_argument('--fine_tune_epoch', default=200, type=int, help='total number of fine tuning epochs')
parser.add_argument('--para_lr_fine_tune', default=[5e-6, 5e-7, 5e-8, 5e-9], type=float, nargs='+',
                    help='step size of parameter update for fine-tuning stage')
# prior setting
parser.add_argument('--sigma0', default=0.0001, type=float, help='sigma_0^2 in prior')
parser.add_argument('--sigma1', default=0.01, type=float, help='sigma_1^2 in prior')
parser.add_argument('--lambda_n', default=0.00001, type=float, help='lambda_n in prior')

args = parser.parse_args()


class SimData_1(Dataset):
    """
    generate simulation data
    seed: random seed to generate the dataset
    data_size: the number of data points in the dataset
    """
    def __init__(self, seed, data_size):
        self.data_size = data_size
        self.treat = np.zeros(self.data_size)
        self.x = np.zeros([self.data_size] + [100])
        self.y = np.zeros((self.data_size, 1))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(seed)
        for i in range(self.data_size):
            # generate x
            ee = truncnorm.rvs(-10, 10)
            x_temp = truncnorm.rvs(-10, 10, size=100) + ee
            x_temp /= np.sqrt(2)

            # node in the first hidden layer
            h1 = np.tanh(0.1*x_temp[0]-0.2*x_temp[1])
            h2 = np.tanh(0.1*x_temp[2]-0.3*x_temp[0])
            h3 = np.tanh(0.2*x_temp[2]+0.4*x_temp[3])
            h4 = np.tanh(0.2*x_temp[4])

            # generate treatment
            score = np.tanh(0.2*h1+0.3*h2-0.1*h3) + np.random.normal(0, 1)
            prob = np.exp(score)/(1 + np.exp(score))
            if prob > 0.5:
                treat_temp = 1
            else:
                treat_temp = 0

            # generate outcome variable
            y_temp = np.tanh(0.3 * np.tanh(-0.3 * np.tanh(-0.4 * h1 + 0.2 * h3) + 0.5 * treat_temp) + 0.1 * np.tanh(
                0.1 * h2 - 0.2 * h4)) + np.random.normal(0, 1)

            self.x[i:] = x_temp
            self.treat[i] = treat_temp
            self.y[i] = y_temp

        self.x = torch.FloatTensor(self.x).to(device)
        self.treat = torch.FloatTensor(self.treat).to(device)
        self.y = torch.FloatTensor(self.y).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        return y, treat, x


def optimization(net, train_data, val_data, epochs, para_lrs):
    """
    train the network
    input:
    net: Net object from network.py
        the network to be trained
    train_data: DataLoader object
        training data
    val_data: DataLoader object
        validation data
    epochs: float
        the number of epochs that the network is trained for
    para_lrs:
        learning rate for parameter update

    output:
    para_path: dictionary
            parameter values for each epoch
    para_gamma_path: dictionary
            indicator for connection selection for each epoch
    var_gamma: dictionary
            indicator for variable selection after all the epochs
    performance: dictionary
            model performance
    hidden_likelihood: list
            likelihoods for all hidden layers after all the epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # imputation parameters
    impute_lrs = args.impute_lr
    ita = args.ita
    mh_step = args.mh_step
    sigma_list = args.sigma

    # training parameters
    para_momentum = args.para_momentum
    train_size = args.train_size
    batch_size = args.batch_size

    # prior parameters
    prior_sigma_0 = args.sigma0
    prior_sigma_1 = args.sigma1
    lambda_n = args.lambda_n

    # save parameter values, indicator variables, and selected input
    para_path = {}
    para_gamma_path = {}
    var_gamma = {}
    for name, para in net.named_parameters():
        para_path[name] = {}
        para_gamma_path[name] = {}
        for epoch in range(epochs):
            para_path[name][str(epoch)] = np.zeros(para.shape)
            para_gamma_path[name][str(epoch)] = np.zeros(para.shape)

    # save training and validation loss
    performance = {}
    train_loss_path = []
    val_loss_path = []
    performance['train_loss'] = train_loss_path
    performance['val_loss'] = val_loss_path

    # define optimizer for parameter update
    optimizer_list = []
    for j in range(net.num_hidden + 1):
        optimizer_list.append(SGD(net.module_dict[str(j)].parameters(), lr=para_lrs[j], momentum=para_momentum))

    # save hidden likelihood for to calculate BIC
    hidden_likelihood = np.zeros(net.num_hidden+1)

    # settings for loss functions
    loss = nn.MSELoss()
    loss_sum = nn.MSELoss(reduction='sum')

    # parameter prior
    c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
    c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1

    # threshold for sparsity
    threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
            0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

    for epoch in range(epochs):
        print("Epoch" + str(epoch))
        # tic = time.time()
        # train loop
        for y, treat, x in train_data:
            # initialize momentum term and hidden units
            hidden_list, momentum_list = [], []
            hidden_list.append(net.module_dict[str(0)](x).detach())
            momentum_list.append(torch.zeros_like(hidden_list[-1]))
            for layer_index in range(net.num_hidden - 1):
                hidden_list.append(net.module_dict[str(layer_index + 1)](hidden_list[-1]).detach())
                momentum_list.append(torch.zeros_like(hidden_list[-1]))
                if layer_index + 1 == net.treat_layer:
                    hidden_list[-1][:, net.treat_node] = treat
            for i in range(hidden_list.__len__()):
                hidden_list[i].requires_grad = True

            # backward imputation by SGHMC
            for step in range(mh_step):
                for layer_index in reversed(range(net.num_hidden)):
                    hidden_list[layer_index].grad = None

                    hidden_likelihood1 = net.likelihood(hidden_list, layer_index + 1, loss_sum, sigma_list, x, y)
                    hidden_likelihood2 = net.likelihood(hidden_list, layer_index, loss_sum, sigma_list, x, y)

                    hidden_likelihood1.backward()
                    hidden_likelihood2.backward()

                    alpha = impute_lrs[layer_index] * ita
                    lr = impute_lrs[layer_index]
                    with torch.no_grad():
                        momentum_list[layer_index] = (1 - alpha) * momentum_list[layer_index] + lr * hidden_list[
                            layer_index].grad + torch.FloatTensor(hidden_list[layer_index].shape).to(device).normal_().mul(
                            np.sqrt(2 * alpha * lr))
                        if layer_index == net.treat_layer:
                            # treatment node will not be updated
                            momentum_list[layer_index][:, net.treat_node] = torch.zeros_like(treat)

                        hidden_list[layer_index].data += momentum_list[layer_index]

            # parameter update
            for para in net.parameters():
                para.grad = None

            with torch.no_grad():  # prior gradient, update for each parameter
                for para in net.parameters():
                    temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                    prior_grad = temp.div(train_size)
                    para.grad = -prior_grad

            for layer_index in range(net.num_hidden + 1):  # data likelihood gradient. update layer by layer
                likelihood = net.likelihood(hidden_list, layer_index, loss_sum, sigma_list, x, y)
                hidden_likelihood[layer_index] = likelihood
                likelihood = -likelihood/batch_size
                optimizer = optimizer_list[layer_index]  # note that the optimizer can only update the parameters by layer
                likelihood.backward()
                optimizer.step()

        train_loss = 0
        with torch.no_grad():
            for y, treat, x in train_data:
                pred = net.forward(x, treat)
                train_loss += loss(pred, y).item()

        train_loss /= len(train_data)
        train_loss_path.append(train_loss)
        print(f"Avg train loss: {train_loss:>8f} \n")

        # validation loop
        val_loss = 0
        with torch.no_grad():
            for y, treat, x in val_data:
                pred = net.forward(x, treat)
                val_loss += loss(pred, y).item()

        val_loss /= len(val_data)
        val_loss_path.append(val_loss)
        print(f"Avg val loss: {val_loss:>8f} \n")

        #toc = time.time()
        #print("optimization time", toc-tic)

        # save parameter values and select connections
        for name, para in net.named_parameters():
            para_path[name][str(epoch)] = para.cpu().data.numpy()
            para_gamma_path[name][str(epoch)] = (para.abs() > threshold).cpu().data.numpy()

        # select input variable
        var_ind = para_gamma_path['0.weight'][str(epoch)]
        for i, (name, para) in enumerate(net.named_parameters()):
            if i % 2 == 0 and i > 0:
                var_ind = np.matmul(para_gamma_path[name][str(epoch)], var_ind)
                if i/2 == net.treat_layer:
                    var_ind[net.treat_node, :] = 0
        variable_selected = np.max(var_ind, 0)
        num_selected = np.sum(variable_selected)
        var_gamma[str(epoch)] = variable_selected.tolist()
        print('number of selected input variable:', num_selected)

    return para_path, para_gamma_path, var_gamma, performance, hidden_likelihood


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate dataset
    data_seed = args.data_seed
    partition_seed = args.partition_seed
    train_size = args.train_size
    val_size = args.val_size

    data = SimData_1(data_seed, train_size + val_size)
    train_set, val_set = random_split(data, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(partition_seed))

    # load training data and validation data
    num_workers = args.num_workers
    batch_size = args.batch_size
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # network setup
    num_hidden = args.layer
    treat_depth = args.depth
    hidden_dim = args.unit
    treat_node = args.treat_node
    input_dim = 100
    output_dim = 1

    # set number of independent runs for sparsity
    num_seed = args.num_run

    # training setting
    para_lrs_train = args.para_lr_train
    para_lrs_fine_tune = args.para_lr_fine_tune
    training_epochs = args.train_epoch
    fine_tune_epoch = args.fine_tune_epoch

    # training results containers
    dim_list = np.zeros([num_seed])  # total number of non-zero element of the pruned network
    BIC_list = np.zeros([num_seed])  # BIC value for model selection
    num_selection_list = np.zeros([num_seed])  # number of selected input variables
    train_loss_list = np.zeros([num_seed])
    val_loss_list = np.zeros([num_seed])

    # prior parameters
    prior_sigma_0 = args.sigma0
    prior_sigma_1 = args.sigma1
    lambda_n = args.lambda_n

    # threshold for sparsity
    threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
            0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

    # path to save the result
    base_path = os.path.join('.', 'stonet_sparsity', 'result', 'sim', 'regression')
    spec = str(para_lrs_train) + '_' + str(hidden_dim) + '_' + str(training_epochs) + '_' + str(
        fine_tune_epoch) + '_' + str(data_seed)
    base_path = os.path.join(base_path, spec)

    for prune_seed in range(num_seed):
        # tic = time.time()
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
        net.to(device)

        para_train, para_gamma_train, var_gamma_train, performance_train, _ = optimization(net, train_data, val_data,
                                                                                           training_epochs, para_lrs_train)

        # prune network parameters
        with torch.no_grad():
            for name, para in net.named_parameters():
                para.data = torch.FloatTensor(para_train[name][str(training_epochs-1)]).to(device)

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
            = optimization(net, train_data, val_data,
                           fine_tune_epoch, para_lrs_fine_tune)


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

        # toc = time.time()
        # print("running time for each initialization", toc - tic)

    np.savetxt(os.path.join(base_path, 'Overall_train_loss.txt'), train_loss_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_val_loss.txt'), val_loss_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_BIC.txt'), BIC_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_non_zero_connections.txt'), dim_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_selected_variables.txt'), num_selection_list, fmt="%s")


if __name__ == '__main__':
    main()
