from model.network import StoNet_Causal
from model.training import training
from data.acic_data import acic_data_cont
from torch.utils.data import DataLoader, random_split
from torchtext.data import to_map_style_dataset
import torch
import numpy as np
import argparse
import os
import errno
from torch.optim import SGD
import json

parser = argparse.ArgumentParser(description='Run Causal StoNet for ACIC data (continuous)')
# Basic Setting
# dataset setting
parser.add_argument('--partition_seed', default=1, type=int, help='set seed for dataset partition')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for DataLoader')
parser.add_argument('--acic_dgp', default=1, type=int, help='data generating process number for ACIC data')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# Parameter for StoNet
# model
parser.add_argument('--layer', default=3, type=int, help='number of hidden layer (not including the treatment layer')
parser.add_argument('--unit', default=[6, 4, 3], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[1e-3, 1e-5, 1e-7, 1e-9], type=float, nargs='+',
                    help='variance of each layer for the model')
parser.add_argument('--depth', default=1, type=int, help='number of layers before the treatment layer')
parser.add_argument('--treat_node', default=1, type=int, help='the position of the treatment variable')

# training setting
parser.add_argument('--pretrain_epoch', default=100, type=int, help='total number of pretraining epochs')
parser.add_argument('--train_epoch', default=1500, type=int, help='total number of training epochs')
parser.add_argument('--mh_step', default=1, type=int, help='number of SGHMC step for imputation')
parser.add_argument('--impute_lr', default=[3e-3, 3e-4, 1e-6], type=float, nargs='+', help='step size for SGHMC')
parser.add_argument('--impute_alpha', default=0.1, type=float, help='momentum weight for SGHMC')
parser.add_argument('--para_lr_train', default=[3e-3, 3e-5, 3e-7, 3e-12], type=float, nargs='+',
                    help='step size for parameter update during training stage')
parser.add_argument('--para_momentum', default=0.9, type=float, help='momentum weight for parameter update')
parser.add_argument('--temperature', default=2, type=float, help="temperature parameter for SGHMC")

# Parameters for Sparsity
parser.add_argument('--num_run', default=10, type=int, help='Number of different initialization used to train the model')
parser.add_argument('--fine_tune_epoch', default=200, type=int, help='total number of fine tuning epochs')
parser.add_argument('--para_lr_fine_tune', default=[3e-4, 3e-6, 3e-8, 3e-13], type=float, nargs='+',
                    help='step size of parameter update for fine-tuning stage')
# prior setting
parser.add_argument('--sigma0', default=2e-5, type=float, help='sigma_0^2 in prior')
parser.add_argument('--sigma1', default=1e-2, type=float, help='sigma_1^2 in prior')
parser.add_argument('--lambda_n', default=1e-4, type=float, help='lambda_n in prior')

args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset setting
    dgp = args.acic_dgp
    partition_seed = args.partition_seed
    data = to_map_style_dataset(acic_data_cont(dgp))
    data_size = data.__len__()
    train_size = int(data_size * 0.6)
    val_size = int(data_size * 0.2)
    test_size = int(data_size * 0.2)
    train_set, val_set, test_set = random_split(data, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(partition_seed))

    # load training data and validation data
    num_workers = args.num_workers
    batch_size = args.batch_size
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data = DataLoader(test_set, batch_size=test_size, num_workers=num_workers)

    # network setup
    num_hidden = args.layer
    treat_depth = args.depth
    hidden_dim = args.unit
    treat_node = args.treat_node
    _, _, x_temp = next(iter(train_data))
    input_dim = x_temp[0].size(dim=0)
    output_dim = 1

    # set number of independent runs for sparsity
    num_seed = args.num_run

    # training setting
    para_lrs_train = args.para_lr_train
    para_lrs_fine_tune = args.para_lr_fine_tune
    para_momentum = args.para_momentum
    training_epochs = args.train_epoch
    pretrain_epochs = args.pretrain_epoch
    fine_tune_epochs = args.fine_tune_epoch

    # imputation parameters
    impute_lrs = args.impute_lr
    alpha = args.impute_alpha
    mh_step = args.mh_step
    sigma_list = args.sigma
    temperature = args.temperature

    # prior parameters
    prior_sigma_0 = args.sigma0
    prior_sigma_1 = args.sigma1
    lambda_n = args.lambda_n

    # threshold for sparsity
    threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
            0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

    # training results containers
    dim_list = np.zeros([num_seed])  # total number of non-zero element of the pruned network
    BIC_list = np.zeros([num_seed])  # BIC value for model selection
    num_selection_out_list = np.zeros([num_seed])  # number of selected input for outcome variable
    num_selection_treat_list = np.zeros([num_seed])  # number of selected input for treatment
    train_loss_list = np.zeros([num_seed])
    val_loss_list = np.zeros([num_seed])
    ate_list = np.zeros([num_seed])

    # path to save the result
    base_path = os.path.join('.', 'acic', 'result', str(dgp))
    basic_spec = str(sigma_list) + '_' + str(mh_step) + '_' + str(training_epochs)
    spec = str(impute_lrs) + '_' + str(para_lrs_train) + '_' + str(prior_sigma_0) + '_' + \
           str(prior_sigma_1) + '_' + str(lambda_n)
    base_path = os.path.join(base_path, basic_spec, spec)

    for prune_seed in range(num_seed):
        print('number of runs', prune_seed)

        PATH = os.path.join(base_path, str(prune_seed))
        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        np.random.seed(prune_seed)
        torch.manual_seed(prune_seed)
        net = StoNet_Causal(num_hidden, hidden_dim, input_dim, output_dim, treat_depth, treat_node)
        net.to(device)

        # optimizer
        optimizer_list_train = []
        for i in range(net.num_hidden + 1):
            # set maximize = True to do gradient ascent
            optimizer_list_train.append(SGD(net.module_list[i].parameters(), lr=para_lrs_train[i],
                                            momentum=para_momentum, maximize=True))

        optimizer_list_fine_tune = []
        for j in range(net.num_hidden + 1):
            # set maximize = True to do gradient ascent
            optimizer_list_fine_tune.append(SGD(net.module_list[j].parameters(), lr=para_lrs_fine_tune[j],
                                                momentum=para_momentum, maximize=True))

        optim_args = dict(train_data=train_data, val_data=val_data, batch_size=batch_size, alpha=alpha, mh_step=mh_step,
                          sigma_list=sigma_list, temperature=temperature, prior_sigma_0=prior_sigma_0,
                          prior_sigma_1=prior_sigma_1, lambda_n=lambda_n)
        # pretrain
        print("Pretrain")
        output_pretrain = training(mode="pretrain", net=net, epochs=pretrain_epochs, optimizer_list=optimizer_list_train,
                                   impute_lrs=impute_lrs, **optim_args)
        para_pretrain = output_pretrain["para_path"]
        para_grad_pretrain = output_pretrain["para_grad_path"]
        para_gamma_pretrain = output_pretrain["para_gamma_path"]
        performance_pretrain = output_pretrain["performance"]

        para_gamma_file = open(os.path.join(PATH, 'para_gamma_pretrain.json'), "w")
        json.dump(para_gamma_pretrain, para_gamma_file, indent="")
        para_gamma_file.close()

        para_file = open(os.path.join(PATH, 'para_pretrain.json'), "w")
        json.dump(para_pretrain, para_file, indent="")
        para_file.close()

        para_grad_file = open(os.path.join(PATH, 'para_grad_pretrain.json'), "w")
        json.dump(para_grad_pretrain, para_grad_file, indent="")
        para_grad_file.close()

        performance_file = open(os.path.join(PATH, 'performance_pretrain.json'), "w")
        json.dump(performance_pretrain, performance_file, indent="")
        performance_file.close()

        # train
        print("Train")
        output_train = training(mode="train", net=net, epochs=training_epochs, optimizer_list=optimizer_list_train,
                                impute_lrs=impute_lrs, **optim_args)
        para_train = output_train["para_path"]
        para_grad_train = output_train["para_grad_path"]
        para_gamma_train = output_train["para_gamma_path"]
        var_gamma_out_train = output_train["input_gamma_path"]["var_selected_out"]
        num_gamma_out_train = output_train["input_gamma_path"]["num_selected_out"]
        var_gamma_treat_train = output_train["input_gamma_path"]["var_selected_treat"]
        num_gamma_treat_train = output_train["input_gamma_path"]["num_selected_treat"]
        performance_train = output_train["performance"]
        impute_lrs_fine_tune = output_train["impute_lrs"]

        # prune network parameters
        with torch.no_grad():
            for name, para in net.named_parameters():
                para.data = torch.FloatTensor(para_train[str(training_epochs-1)][name]).to(device)

        user_mask = {}
        for name, para in net.named_parameters():
            user_mask[name] = para.abs() < threshold
        net.set_prune(user_mask)

        # save model training results
        num_selection_out_list[prune_seed] = num_gamma_out_train[training_epochs-1]
        num_selection_treat_list[prune_seed] = num_gamma_treat_train[training_epochs-1]

        temp_str = [str(int(x)) for x in var_gamma_out_train[str(training_epochs-1)]]
        temp_str = ' '.join(temp_str)
        filename = PATH + 'selected_variable_out.txt'
        f = open(filename, 'w')
        f.write(temp_str)
        f.close()

        temp_str = [str(int(x)) for x in var_gamma_treat_train[str(training_epochs-1)]]
        temp_str = ' '.join(temp_str)
        filename = PATH + 'selected_variable_treat.txt'
        f = open(filename, 'w')
        f.write(temp_str)
        f.close()

        para_gamma_file = open(os.path.join(PATH, 'para_gamma_train.json'), "w")
        json.dump(para_gamma_train, para_gamma_file, indent="")
        para_gamma_file.close()

        para_file = open(os.path.join(PATH, 'para_train.json'), "w")
        json.dump(para_train, para_file, indent="")
        para_file.close()

        para_grad_file = open(os.path.join(PATH, 'para_grad_train.json'), "w")
        json.dump(para_grad_train, para_grad_file, indent="")
        para_grad_file.close()

        performance_file = open(os.path.join(PATH, 'performance_train.json'), "w")
        json.dump(performance_train, performance_file, indent="")
        performance_file.close()

        var_gamma_file = open(os.path.join(PATH, 'var_gamma_out_train.json'), "w")
        json.dump(var_gamma_out_train, var_gamma_file, indent="")
        var_gamma_file.close()

        num_gamma_file = open(os.path.join(PATH, 'num_selected_out_train.json'), "w")
        json.dump(num_gamma_out_train, num_gamma_file, indent="")
        num_gamma_file.close()

        var_gamma_file = open(os.path.join(PATH, 'var_gamma_treat_train.json'), "w")
        json.dump(var_gamma_treat_train, var_gamma_file, indent="")
        var_gamma_file.close()

        num_gamma_file = open(os.path.join(PATH, 'num_selected_treat_train.json'), "w")
        json.dump(num_gamma_treat_train, num_gamma_file, indent="")
        num_gamma_file.close()

        # refine non-zero network parameters
        print("Refine Weight")
        output_fine_tune = training(mode="train", net=net, epochs=fine_tune_epochs, optimizer_list=optimizer_list_fine_tune,
                                    impute_lrs=impute_lrs_fine_tune, **optim_args)
        para_fine_tune = output_fine_tune["para_path"]
        para_grad_fine_tune = output_train["para_grad_path"]
        para_gamma_fine_tune = output_fine_tune["para_gamma_path"]
        var_gamma_out_fine_tune = output_fine_tune["input_gamma_path"]["var_selected_out"]
        num_gamma_out_fine_tune = output_fine_tune["input_gamma_path"]["num_selected_out"]
        var_gamma_treat_fine_tune = output_fine_tune["input_gamma_path"]["var_selected_treat"]
        num_gamma_treat_fine_tune = output_fine_tune["input_gamma_path"]["num_selected_treat"]
        performance_fine_tune = output_fine_tune["performance"]

        # save fine tuning results
        para_gamma_file = open(os.path.join(PATH, 'para_gamma_fine_tune.json'), "w")
        json.dump(para_gamma_fine_tune, para_gamma_file, indent="")
        para_gamma_file.close()

        para_file = open(os.path.join(PATH, 'para_fine_tune.json'), "w")
        json.dump(para_fine_tune, para_file, indent="")
        para_file.close()

        para_grad_file = open(os.path.join(PATH, 'para_grad_fine_tune.json'), "w")
        json.dump(para_grad_fine_tune, para_grad_file, indent="")
        para_grad_file.close()

        performance_file = open(os.path.join(PATH, 'performance_fine_tune.json'), "w")
        json.dump(performance_fine_tune, performance_file, indent="")
        performance_file.close()

        var_gamma_file = open(os.path.join(PATH, 'var_gamma_out_fine_tune.json'), "w")
        json.dump(var_gamma_out_fine_tune, var_gamma_file, indent="")
        var_gamma_file.close()

        num_gamma_file = open(os.path.join(PATH, 'num_selected_out_fine_tune.json'), "w")
        json.dump(num_gamma_out_fine_tune, num_gamma_file, indent="")
        num_gamma_file.close()

        var_gamma_file = open(os.path.join(PATH, 'var_gamma_treat_fine_tune.json'), "w")
        json.dump(var_gamma_treat_fine_tune, var_gamma_file, indent="")
        var_gamma_file.close()

        num_gamma_file = open(os.path.join(PATH, 'num_selected_treat_fine_tune.json'), "w")
        json.dump(num_gamma_treat_fine_tune, num_gamma_file, indent="")
        num_gamma_file.close()

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

            BIC = (train_size * train_loss + np.log(train_size) *num_non_zero_element).item()
            BIC_list[prune_seed] = BIC

            print("number of non-zero connections:", num_non_zero_element)
            print('BIC:', BIC)

        # calculate doubly-estimator of ate
        with torch.no_grad():
            ate_db = 0  # doubly-robust estimate of average treatment effect
            for y, treat, x in test_data:
                pred, prop_score = net.forward(x, treat)
                counter_fact, _ = net.forward(x, 1 - treat)
                outcome_contrast = torch.flatten(pred-counter_fact) * (2*treat - 1)
                prop_contrast = treat/prop_score - (1-treat)/(1-prop_score)
                pred_resid = torch.flatten(y - pred)
                ate_db = torch.mean(outcome_contrast + prop_contrast * pred_resid)

            ate_list[prune_seed] = ate_db

        torch.save(net.state_dict(), os.path.join(PATH, 'model' + str(prune_seed)+'.pt'))

    np.savetxt(os.path.join(base_path, 'Overall_train_loss.txt'), train_loss_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_val_loss.txt'), val_loss_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_BIC.txt'), BIC_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_non_zero_connections.txt'), dim_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_selected_variables_out.txt'), num_selection_out_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_selected_variables_treat.txt'), num_selection_treat_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_ATE.txt'), ate_list, fmt="%s")


if __name__ == '__main__':
    main()