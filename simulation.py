from model.network import StoNet_Causal
from model.training import training
from data import Simulation1_complete
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch
import numpy as np
import argparse
import os
import errno
from torch.optim import SGD
from pickle import dump

parser = argparse.ArgumentParser(description='Run Simulation for Causal StoNet')
# Basic Setting
# simulation setting
parser.add_argument('--data_seed', default=1, type=int, help='set seed for data generation')
parser.add_argument('--partition_seed', default=1, type=int, help='set seed for dataset partition')

# dataset setting
parser.add_argument('--train_size', default=10000, type=int, help='size of training set')
parser.add_argument('--val_size', default=1000, type=int, help='size of validation set')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# Parameter for StoNet
# model
parser.add_argument('--layer', default=3, type=int, help='number of hidden layers')
parser.add_argument('--unit', default=[6, 4, 3], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[1e-3, 1e-5, 1e-7, 1e-9], type=float, nargs='+',
                    help='variance of each layer for the model')
parser.add_argument('--depth', default=1, type=int, help='number of layers before the treatment layer')
parser.add_argument('--treat_node', default=1, type=int, help='the position of the treatment variable')
parser.add_argument('--treat_loss_weight', default=1000, type=float, help='weight for the treatment loss')
parser.add_argument('--regression', dest='classification_flag', action='store_false', help='false for regression')
parser.add_argument('--classification', dest='classification_flag', action='store_true', help='true for classification')

# training setting
parser.add_argument('--pretrain_epoch', default=100, type=int, help='total number of pretraining epochs')
parser.add_argument('--train_epoch', default=1500, type=int, help='total number of training epochs')
parser.add_argument('--mh_step', default=1, type=int, help='number of SGHMC step for imputation')
parser.add_argument('--impute_lr', default=[3e-3, 3e-4, 1e-6], type=float, nargs='+', help='step size for SGHMC')
parser.add_argument('--impute_alpha', default=0.1, type=float, help='momentum weight for SGHMC')
parser.add_argument('--para_lr_train', default=[3e-4, 3e-6, 3e-8, 1e-12], type=float, nargs='+',
                    help='step size for parameter update during training stage')
parser.add_argument('--para_momentum', default=0.9, type=float, help='momentum weight for parameter update')
parser.add_argument('--para_lr_decay', default=1.2, type=float, help='decay factor for para_lr')
parser.add_argument('--impute_lr_decay', default=1, type=float, help='decay factor for impute_lr')

# Parameters for Sparsity
parser.add_argument('--num_run', default=10, type=int, help='Number of different initialization used to train the model')
parser.add_argument('--fine_tune_epoch', default=200, type=int, help='total number of fine tuning epochs')
parser.add_argument('--para_lr_fine_tune', default=[3e-5, 3e-7, 3e-9, 1e-13], type=float, nargs='+',
                    help='step size of parameter update for fine-tuning stage')
# prior setting
parser.add_argument('--sigma0', default=1e-3, type=float, help='sigma_0^2 in prior')
parser.add_argument('--sigma1', default=1e-1, type=float, help='sigma_1^2 in prior')
parser.add_argument('--lambda_n', default=1e-6, type=float, help='lambda_n in prior')

args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # task
    classification_flag = args.classification_flag

    # generate dataset
    data_seed = args.data_seed
    train_size = args.train_size
    val_size = args.val_size
    data = Simulation1_complete(data_seed)
    train_set, val_set, test_set = random_split(data, [train_size, val_size, val_size],
                                                generator=torch.Generator().manual_seed(1))
    in_sample_set = ConcatDataset([train_set, val_set])

    # load training data and validation data
    batch_size = args.batch_size
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    in_sample_data = DataLoader(in_sample_set, batch_size=batch_size)

    # network setup
    net_args = dict(num_hidden=args.layer, hidden_dim=args.unit, input_dim=data.x[0].size(dim=0),
                    output_dim=len(data.y.unique()) if classification_flag else 1,
                    treat_layer=args.depth, treat_node=args.treat_node)

    # set number of independent runs for sparsity
    num_seed = args.num_run

    # training setting
    para_lrs_train = args.para_lr_train
    para_lrs_fine_tune = args.para_lr_fine_tune
    para_momentum = args.para_momentum
    training_epochs = args.train_epoch
    pretrain_epochs = args.pretrain_epoch
    fine_tune_epochs = args.fine_tune_epoch
    para_lr_decay = args.para_lr_decay
    impute_lr_decay = args.impute_lr_decay
    treat_loss_weight = args.treat_loss_weight

    # imputation parameters
    impute_lrs = args.impute_lr
    mh_step = args.mh_step
    sigma_list = args.sigma

    # prior parameters
    prior_sigma_0 = args.sigma0
    prior_sigma_1 = args.sigma1
    lambda_n = args.lambda_n

    # threshold for sparsity
    threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
            0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

    # training results containers
    results = dict(dim=0, BIC=0, num_selection_out=0, num_selection_treat=0, out_train_loss=0, out_val_loss=0,
                   treat_train_loss=0, treat_val_loss=0, treat_train_acc=0, treat_val_acc=0,
                   error_in_sample=0, error_out_sample=0)
    BIC_list = []  # BIC value for model selection
    if classification_flag:
        results.update([('out_train_acc', 0), ('out_val_acc', 0)])

    # path to save the result
    base_path = os.path.join('.', 'simulation1', 'result', str(data_seed))
    basic_spec = str(sigma_list) + '_' + str(mh_step) + '_' + str(training_epochs) + '_' + str(treat_loss_weight)
    spec = str(impute_lrs) + '_' + str(para_lrs_train) + '_' + str(prior_sigma_0) + '_' + \
           str(prior_sigma_1) + '_' + str(lambda_n)
    decay_spec = str(impute_lr_decay) + '_' + str(para_lr_decay)
    base_path = os.path.join(base_path, basic_spec, spec, decay_spec)

    for prune_seed in range(num_seed):
        print('number of runs', prune_seed)

        PATH = os.path.join(base_path)
        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        # initialize network
        np.random.seed(prune_seed)
        torch.manual_seed(prune_seed)
        net = StoNet_Causal(**net_args)
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

        optim_args = dict(train_data=train_data, val_data=val_data, batch_size=batch_size, alpha=args.impute_alpha,
                          mh_step=mh_step, sigma_list=sigma_list, prior_sigma_0=prior_sigma_0,
                          prior_sigma_1=prior_sigma_1, lambda_n=lambda_n, para_lr_decay=para_lr_decay,
                          impute_lr_decay=impute_lr_decay, outcome_cat=classification_flag,
                          treat_loss_weight=treat_loss_weight)
        # pretrain
        print("Pretrain")
        output_pretrain = training(mode="pretrain", net=net, epochs=pretrain_epochs, optimizer_list=optimizer_list_train,
                                   impute_lrs=impute_lrs, **optim_args)
        # para_pretrain = output_pretrain["para_path"]
        # para_grad_pretrain = output_pretrain["para_grad_path"]
        # para_gamma_pretrain = output_pretrain["para_gamma_path"]
        performance_pretrain = output_pretrain["performance"]

        # with open(os.path.join(PATH, 'para_gamma_pretrain.pkl'), "wb") as f:
        #     dump(para_gamma_pretrain, f)

        # with open(os.path.join(PATH, 'para_pretrain.pkl'), "wb") as f:
        #     dump(para_pretrain, f)

        # with open(os.path.join(PATH, 'para_grad_pretrain.pkl'), "wb") as f:
        #     dump(para_grad_pretrain, f)

        with open(os.path.join(PATH, 'performance_pretrain.pkl'), "wb") as f:
            dump(performance_pretrain, f)

        # train
        print("Train")
        output_train = training(mode="train", net=net, epochs=training_epochs, optimizer_list=optimizer_list_train,
                                impute_lrs=impute_lrs, **optim_args)
        para_train = output_train["para_path"]
        # para_grad_train = output_train["para_grad_path"]
        # para_gamma_train = output_train["para_gamma_path"]
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
        net.prune_masked_para()

        # save model training results
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

        # with open(os.path.join(PATH, 'para_gamma_train.pkl'), "wb") as f:
        #     dump(para_gamma_train, f)

        # with open(os.path.join(PATH, 'para_train.pkl'), "wb") as f:
        #     dump(para_train, f)

        # with open(os.path.join(PATH, 'para_grad_train.pkl'), "wb") as f:
        #     dump(para_grad_train, f)

        with open(os.path.join(PATH, 'performance_train.pkl'), "wb") as f:
            dump(performance_train, f)

        # with open(os.path.join(PATH, 'var_gamma_out_train.pkl'), "wb") as f:
        #     dump(var_gamma_out_train, f)

        # with open(os.path.join(PATH, 'num_selected_out_train.pkl'), "wb") as f:
        #     dump(num_gamma_out_train, f)

        # with open(os.path.join(PATH, 'var_gamma_treat_train.pkl'), "wb") as f:
        #     dump(var_gamma_treat_train, f)

        # with open(os.path.join(PATH, 'num_selected_treat_train.pkl'), "wb") as f:
        #     dump(num_gamma_treat_train, f)

        # refine non-zero network parameters
        print("Refine Weight")
        output_fine_tune = training(mode="train", net=net, epochs=fine_tune_epochs, optimizer_list=optimizer_list_fine_tune,
                                    impute_lrs=impute_lrs_fine_tune, **optim_args)
        # para_fine_tune = output_fine_tune["para_path"]
        # para_grad_fine_tune = output_fine_tune["para_grad_path"]
        # para_gamma_fine_tune = output_fine_tune["para_gamma_path"]
        # var_gamma_out_fine_tune = output_fine_tune["input_gamma_path"]["var_selected_out"]
        # num_gamma_out_fine_tune = output_fine_tune["input_gamma_path"]["num_selected_out"]
        # var_gamma_treat_fine_tune = output_fine_tune["input_gamma_path"]["var_selected_treat"]
        # num_gamma_treat_fine_tune = output_fine_tune["input_gamma_path"]["num_selected_treat"]
        performance_fine_tune = output_fine_tune["performance"]
        likelihoods = output_fine_tune["likelihoods"]

        # save fine tuning results
        # with open(os.path.join(PATH, 'para_gamma_fine_tune.pkl'), "wb") as f:
        #     dump(para_gamma_fine_tune, f)

        # with open(os.path.join(PATH, 'para_fine_tune.pkl'), "wb") as f:
        #     dump(para_fine_tune, f)

        # with open(os.path.join(PATH, 'para_grad_fine_tune.pkl'), "wb") as f:
        #     dump(para_grad_fine_tune, f)

        with open(os.path.join(PATH, 'performance_fine_tune.pkl'), "wb") as f:
            dump(performance_fine_tune, f)

        # with open(os.path.join(PATH, 'var_gamma_out_fine_tune.pkl'), "wb") as f:
        #     dump(var_gamma_out_fine_tune, f)

        # with open(os.path.join(PATH, 'num_selected_out_fine_tune.pkl'), "wb") as f:
        #     dump(num_gamma_out_fine_tune, f)

        # with open(os.path.join(PATH, 'var_gamma_treat_fine_tune.pkl'), "wb") as f:
        #     dump(var_gamma_treat_fine_tune, f)

        # with open(os.path.join(PATH, 'num_selected_treat_fine_tune.pkl'), "wb") as f:
        #     dump(num_gamma_treat_fine_tune, f)

        # calculate BIC
        with torch.no_grad():
            num_non_zero_element = 0
            for name, para in net.named_parameters():
                num_non_zero_element = num_non_zero_element + para.numel() - net.mask_prune[name].sum()

            BIC = (np.log(train_set.__len__()) * num_non_zero_element - 2 * np.sum(likelihoods)).item()
            BIC_list.append(BIC)

            print("number of non-zero connections:", num_non_zero_element)
            print('BIC:', BIC)

        # calculate doubly-robust estimator of ate
        with torch.no_grad():
            ate_est_in, ate_true_in, ate_est_out, ate_true_out = 0, 0, 0, 0
            for y, treat, x, tau in test_data:  # out-of-sample error
                pred, prop_score = net.forward(x, treat)
                counter_fact, _ = net.forward(x, 1 - treat)
                outcome_contrast = torch.flatten(pred-counter_fact) * (2*treat - 1)
                prop_contrast = treat/prop_score - (1-treat)/(1-prop_score)
                pred_resid = torch.flatten(y - pred)
                ate_est_out += torch.sum(outcome_contrast + prop_contrast * pred_resid)
                ate_true_out += torch.sum(tau)
            error_out = abs((ate_est_out-ate_true_out)/len(test_set)).item()

            for y, treat, x, tau in in_sample_data:  # in-sample error
                pred, prop_score = net.forward(x, treat)
                counter_fact, _ = net.forward(x, 1 - treat)
                outcome_contrast = torch.flatten(pred-counter_fact) * (2*treat - 1)
                prop_contrast = treat/prop_score - (1-treat)/(1-prop_score)
                pred_resid = torch.flatten(y - pred)
                ate_est_in += torch.sum(outcome_contrast + prop_contrast * pred_resid)
                ate_true_in += torch.sum(tau)
            error_in = abs((ate_est_in-ate_true_in)/(train_size+val_size)).item()

        if BIC == min(BIC_list):
            results['num_selection_out'] = num_gamma_out_train[training_epochs-1].item()
            results['num_selection_treat'] = num_gamma_treat_train[training_epochs-1].item()
            results['out_train_loss'] = performance_fine_tune['out_train_loss'][-1]
            results['treat_train_loss'] = performance_fine_tune['treat_train_loss'][-1]
            results['out_val_loss'] = performance_fine_tune['out_val_loss'][-1]
            results['treat_val_loss'] = performance_fine_tune['treat_val_loss'][-1]
            results['treat_train_acc'] = performance_fine_tune['treat_train_acc'][-1]
            results['treat_val_acc'] = performance_fine_tune['treat_val_acc'][-1]
            results['error_in_sample'] = error_in
            results['error_out_sample'] = error_out
            if classification_flag:
                results['out_train_acc'] = performance_fine_tune['out_train_acc'][-1]
                results['out_val_acc'] = performance_fine_tune['out_val_acc'][-1]
            results['dim'] = num_non_zero_element.item()
            results['BIC'] = BIC

            torch.save(net.state_dict(), os.path.join(base_path, 'model.pt'))

    # save overall performance
    with open(os.path.join(base_path, 'causal_stoNet_results.pkl'), "wb") as f:
        dump(results, f)


if __name__ == '__main__':
    main()
