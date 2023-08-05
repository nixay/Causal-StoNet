from model.network import StoNet_Causal
from model.training import training
from data import acic_bench
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
import os
import errno
from torch.optim import SGD
from pickle import dump
from sklearn.utils import class_weight
import torch.nn as nn

parser = argparse.ArgumentParser(description='Run Causal StoNet for benchmarking against ACIC')
# Basic Setting
# dataset setting
parser.add_argument('--partition_seed', default=1, type=int, help='set seed for dataset partition')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for DataLoader')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# Parameter for StoNet
# model
parser.add_argument('--layer', default=2, type=int, help='number of hidden layer (not including the treatment layer')
parser.add_argument('--unit', default=[64, 32], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[1e-5, 1e-7, 1e-9], type=float, nargs='+',
                    help='variance of each layer for the model')
parser.add_argument('--depth', default=1, type=int, help='number of layers before the treatment layer')
parser.add_argument('--treat_node', default=1, type=int, help='the position of the treatment variable')
parser.add_argument('--treat_loss_weight', default=2000, type=float, help='weight for the treatment loss')
parser.add_argument('--regression', dest='classification_flag', action='store_false', help='false for regression')
parser.add_argument('--classification', dest='classification_flag', action='store_true', help='true for classification')

# training setting
parser.add_argument('--pretrain_epoch', default=100, type=int, help='total number of pretraining epochs')
parser.add_argument('--train_epoch', default=500, type=int, help='total number of training epochs')
parser.add_argument('--mh_step', default=1, type=int, help='number of SGHMC step for imputation')
parser.add_argument('--impute_lr', default=[3e-4, 1e-6], type=float, nargs='+', help='step size for SGHMC')
parser.add_argument('--impute_alpha', default=0.1, type=float, help='momentum weight for SGHMC')
parser.add_argument('--para_lr_train', default=[5e-6, 5e-8, 5e-13], type=float, nargs='+',
                    help='step size for parameter update during training stage')
parser.add_argument('--para_momentum', default=0.9, type=float, help='momentum weight for parameter update')
parser.add_argument('--para_lr_decay', default=2, type=float, help='decay factor for para_lr')
parser.add_argument('--impute_lr_decay', default=1, type=float, help='decay factor for impute_lr')

# Parameters for Sparsity
parser.add_argument('--num_run', default=10, type=int, help='Number of different initialization used to train the model')
parser.add_argument('--fine_tune_epoch', default=500, type=int, help='total number of fine tuning epochs')
parser.add_argument('--para_lr_fine_tune', default=[5e-7, 1e-8, 1e-13], type=float, nargs='+',
                    help='step size of parameter update for fine-tuning stage')
# prior setting
parser.add_argument('--sigma0', default=1e-4, type=float, help='sigma_0^2 in prior')
parser.add_argument('--sigma1', default=1e-1, type=float, help='sigma_1^2 in prior')
parser.add_argument('--lambda_n', default=1e-6, type=float, help='lambda_n in prior')

args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # task
    classification_flag = args.classification_flag

    # dataset setting
    train_set = acic_bench('train')
    val_set = acic_bench('test')
    # class_weights_out = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_set.y),
    #                                                       y=train_set.y.numpy())
    # class_weights_out = torch.tensor(class_weights_out, dtype=torch.float)
    # class_weights_treat = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_set.treat),
    #                                                       y=train_set.treat.numpy())
    # class_weights_treat = torch.tensor(class_weights_treat, dtype=torch.float)

    # load training data and validation data
    num_workers = args.num_workers
    batch_size = args.batch_size
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # network setup
    _, _, x_temp = next(iter(train_set))
    net_args = dict(num_hidden=args.layer, hidden_dim=args.unit, input_dim=x_temp.size(dim=0),
                    output_dim=len(train_set.y.unique()) if classification_flag else 1,
                    treat_layer=args.depth, treat_node=args.treat_node)

    # number of independent runs for sparsity
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
    dim_list = np.zeros([num_seed])  # total number of non-zero parameters of the pruned network
    BIC_list = np.zeros([num_seed])  # BIC
    num_selection_out_list = np.zeros([num_seed])  # number of selected input for outcome variable
    num_selection_treat_list = np.zeros([num_seed])  # number of selected input for treatment
    out_train_loss_list = np.zeros([num_seed])
    out_val_loss_list = np.zeros([num_seed])
    if classification_flag:
        out_train_acc_list = np.zeros([num_seed])
        out_val_acc_list = np.zeros([num_seed])
    treat_train_loss_list = np.zeros([num_seed])
    treat_val_loss_list = np.zeros([num_seed])
    treat_train_acc_list = np.zeros([num_seed])
    treat_val_acc_list = np.zeros([num_seed])
    ate_list = np.zeros([num_seed])  # estimated average treatment effect

    # path to save the result
    base_path = os.path.join('.', 'acic_bench', 'result')
    basic_spec = str(sigma_list) + '_' + str(mh_step) + '_' + str(training_epochs) + '_' + str(treat_loss_weight)
    spec = str(impute_lrs) + '_' + str(para_lrs_train) + '_' + str(prior_sigma_0) + '_' + \
           str(prior_sigma_1) + '_' + str(lambda_n)
    decay_spec = str(impute_lr_decay) + '_' + str(para_lr_decay)
    base_path = os.path.join(base_path, basic_spec, spec, decay_spec)

    # Training starts here
    for prune_seed in range(num_seed):
        print('number of runs', prune_seed)

        # create the path to save model results
        PATH = os.path.join(base_path, str(prune_seed))
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

        # define optimizer
        optimizer_list_train1 = []
        for i in range(net.num_hidden + 1):
            # set maximize = True to do gradient ascent
            optimizer_list_train1.append(SGD(net.module_list[i].parameters(), lr=para_lrs_train[i],
                                             momentum=para_momentum, maximize=True))

        optimizer_list_train2 = []
        for l in range(net.num_hidden + 1):
            for k in range(training_epochs):
                para_lrs_train_temp = para_lrs_train[l]/(1+para_lrs_train[l]*k**para_lr_decay)
            # set maximize = True to do gradient ascent
            optimizer_list_train2.append(SGD(net.module_list[l].parameters(), lr=para_lrs_train_temp/2,
                                             momentum=para_momentum, maximize=True))

        optimizer_list_fine_tune = []
        for j in range(net.num_hidden + 1):
            # set maximize = True to do gradient ascent
            optimizer_list_fine_tune.append(SGD(net.module_list[j].parameters(), lr=para_lrs_fine_tune[j],
                                                momentum=para_momentum, maximize=True))

        # parameters for training
        optim_args = dict(train_data=train_data, val_data=val_data, batch_size=batch_size, alpha=args.impute_alpha,
                          mh_step=mh_step, sigma_list=sigma_list, prior_sigma_0=prior_sigma_0,
                          prior_sigma_1=prior_sigma_1, lambda_n=lambda_n,
                          para_lr_decay=para_lr_decay, impute_lr_decay=impute_lr_decay, outcome_cat=classification_flag,
                          treat_loss_weight=treat_loss_weight, CE_weight=None)

        # pretrain
        print("Pretrain")
        output_pretrain = training(mode="pretrain", net=net, epochs=pretrain_epochs, optimizer_list=optimizer_list_train1,
                                   impute_lrs=impute_lrs, **optim_args)
        # para_pretrain = output_pretrain["para_path"]
        performance_pretrain = output_pretrain["performance"]

        # with open(os.path.join(PATH, 'para_pretrain.pkl'), "wb") as f:
        #     dump(para_pretrain, f)

        with open(os.path.join(PATH, 'performance_pretrain.pkl'), "wb") as f:
            dump(performance_pretrain, f)

        # train
        print("Train 1")
        output_train1 = training(mode="train", net=net, epochs=training_epochs, optimizer_list=optimizer_list_train1,
                                 impute_lrs=impute_lrs, **optim_args)
        para_train1 = output_train1["para_path"]
        # num_gamma_out_train1 = output_train1["input_gamma_path"]["num_selected_out"]
        # num_gamma_treat_train1 = output_train1["input_gamma_path"]["num_selected_treat"]
        performance_train1 = output_train1["performance"]
        impute_lrs_train2 = output_train1["impute_lrs"]

        # prune network parameters
        with torch.no_grad():
            for name, para in net.named_parameters():
                para.data = torch.FloatTensor(para_train1[str(training_epochs-1)][name]).to(device)

        user_mask = {}
        for name, para in net.named_parameters():
            user_mask[name] = para.abs() < threshold
        net.set_prune(user_mask)
        net.prune_masked_para()

        # save training result
        with open(os.path.join(PATH, 'performance_train1.pkl'), "wb") as f:
            dump(performance_train1, f)

        # with open(os.path.join(PATH, 'num_selected_out_train1.pkl'), "wb") as f:
        #     dump(num_gamma_out_train1, f)

        # with open(os.path.join(PATH, 'num_selected_treat_train1.pkl'), "wb") as f:
        #     dump(num_gamma_treat_train1, f)

        print("Train 2")
        output_train = training(mode="train", net=net, epochs=training_epochs*2, optimizer_list=optimizer_list_train2,
                                impute_lrs=impute_lrs_train2, **optim_args)
        para_train = output_train["para_path"]
        var_gamma_out_train = output_train["input_gamma_path"]["var_selected_out"]
        num_gamma_out_train = output_train["input_gamma_path"]["num_selected_out"]
        var_gamma_treat_train = output_train["input_gamma_path"]["var_selected_treat"]
        num_gamma_treat_train = output_train["input_gamma_path"]["num_selected_treat"]
        performance_train = output_train["performance"]
        impute_lrs_fine_tune = output_train["impute_lrs"]

        # prune network parameters
        with torch.no_grad():
            for name, para in net.named_parameters():
                para.data = torch.FloatTensor(para_train[str(training_epochs*2-1)][name]).to(device)

        user_mask = {}
        for name, para in net.named_parameters():
            user_mask[name] = para.abs() < threshold
        net.set_prune(user_mask)
        net.prune_masked_para()

        # save model training results
        num_selection_out_list[prune_seed] = num_gamma_out_train[training_epochs*2-1]
        num_selection_treat_list[prune_seed] = num_gamma_treat_train[training_epochs*2-1]

        temp_str = [str(int(x)) for x in var_gamma_out_train[str(training_epochs*2-1)]]
        temp_str = ' '.join(temp_str)
        filename = PATH + 'selected_variable_out.txt'
        f = open(filename, 'w')
        f.write(temp_str)
        f.close()

        temp_str = [str(int(x)) for x in var_gamma_treat_train[str(training_epochs*2-1)]]
        temp_str = ' '.join(temp_str)
        filename = PATH + 'selected_variable_treat.txt'
        f = open(filename, 'w')
        f.write(temp_str)
        f.close()

        # with open(os.path.join(PATH, 'para_train.pkl'), "wb") as f:
        #     dump(para_train, f)

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
        para_fine_tune = output_fine_tune["para_path"]
        # var_gamma_out_fine_tune = output_fine_tune["input_gamma_path"]["var_selected_out"]
        # num_gamma_out_fine_tune = output_fine_tune["input_gamma_path"]["num_selected_out"]
        # var_gamma_treat_fine_tune = output_fine_tune["input_gamma_path"]["var_selected_treat"]
        # num_gamma_treat_fine_tune = output_fine_tune["input_gamma_path"]["num_selected_treat"]
        performance_fine_tune = output_fine_tune["performance"]
        likelihoods = output_fine_tune["likelihoods"]

        # save refining results
        # with open(os.path.join(PATH, 'para_fine_tune.pkl'), "wb") as f:
        #     dump(para_fine_tune, f)

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

        # save training results for the final run
        # will need to transfer the losses back to the original scale
        out_train_loss_list[prune_seed] = performance_fine_tune['out_train_loss'][-1]
        out_val_loss_list[prune_seed] = performance_fine_tune['out_val_loss'][-1]
        if classification_flag:
            out_train_acc_list[prune_seed] = performance_fine_tune['out_train_acc'][-1]
            out_val_acc_list[prune_seed] = performance_fine_tune['out_val_acc'][-1]

        treat_train_loss_list[prune_seed] = performance_fine_tune['treat_train_loss'][-1]
        treat_val_loss_list[prune_seed] = performance_fine_tune['treat_val_loss'][-1]
        treat_train_acc_list[prune_seed] = performance_fine_tune['treat_train_acc'][-1]
        treat_val_acc_list[prune_seed] = performance_fine_tune['treat_val_acc'][-1]

        # calculate non-zero connections and BIC
        with torch.no_grad():
            num_non_zero_element = 0
            for name, para in net.named_parameters():
                num_non_zero_element = num_non_zero_element + para.numel() - net.mask_prune[name].sum()
            dim_list[prune_seed] = num_non_zero_element

            BIC = (np.log(train_set.__len__()) * num_non_zero_element - 2 * np.sum(likelihoods)).item()
            BIC_list[prune_seed] = BIC

            print("number of non-zero connections:", num_non_zero_element.item())
            print('BIC:', BIC)

        # calculate doubly-robust estimator of ate: outcome variable is continuous; scaling back is needed
        m = nn.Softmax(dim=1)
        with torch.no_grad():
            ate_db = 0  # doubly-robust estimate of average treatment effect
            for y, treat, x in val_data:
                pred_score, prop_score = net.forward(x, treat)
                pred_prob = m(pred_score)[:, 1]  # E[Y|X, A] = P(Y=1|X, A)
                counter_fact_score, _ = net.forward(x, 1 - treat)
                counter_fact_prob = m(counter_fact_score)[:, 1]
                outcome_contrast = torch.flatten(pred_prob-counter_fact_prob) * (2*treat - 1)
                prop_contrast = treat/prop_score - (1-treat)/(1-prop_score)
                pred_resid = torch.flatten(y - pred_prob)
                ate_db += torch.sum(outcome_contrast + prop_contrast * pred_resid).item()

            ate_list[prune_seed] = ate_db/len(val_set)

        torch.save(net.state_dict(), os.path.join(PATH, 'model' + str(prune_seed)+'.pt'))

    # save overall performance
    np.savetxt(os.path.join(base_path, 'Overall_train_loss_out.txt'), out_train_loss_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_val_loss_out.txt'), out_val_loss_list, fmt="%s")
    if classification_flag:
        np.savetxt(os.path.join(base_path, 'Overall_train_acc_out.txt'), out_train_acc_list, fmt="%s")
        np.savetxt(os.path.join(base_path, 'Overall_val_acc_out.txt'), out_val_acc_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_train_loss_treat.txt'), treat_train_loss_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_val_loss_treat.txt'), treat_val_loss_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_train_acc_treat.txt'), treat_train_acc_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_val_acc_treat.txt'), treat_val_acc_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_BIC.txt'), BIC_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_non_zero_connections.txt'), dim_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_selected_variables_out.txt'), num_selection_out_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_selected_variables_treat.txt'), num_selection_treat_list, fmt="%s")
    np.savetxt(os.path.join(base_path, 'Overall_ATE.txt'), ate_list, fmt="%s")


if __name__ == '__main__':
    main()
