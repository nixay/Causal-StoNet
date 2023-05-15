import numpy as np
import torch
import torch.nn as nn
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


def training(mode, net, train_data, val_data, epochs, batch_size, optimizer_list, impute_lrs, alpha, mh_step,
             sigma_list, prior_sigma_0, prior_sigma_1, lambda_n, para_lr_decay,
             impute_lr_decay, temperature, scalar_y=1, outcome_cat=False, CE_weight=False):

    """
    train the network
    inputs:
    mode: training mode
        for "pretrain", the impute_lr and para_lr will keep constant; pruning result will not be recorded.
        for "train", the impute_lr and para_lr decays epoch by epoch; pruning result will be recorded.
    net: StoNet_Causal object defined in network.py
        the network to be trained
    train_data, val_data: DataLoader object
        training data and validation data, respectively
    epochs: float
        the number of training epochs
    batch_size: int
        sample size of each batch
    optimizer_list: list of torch.optim objects
        optimizers for parameter update
    impute_lrs: list of floats
        learning rate for SGHMC
    alpha: float
        coefficient for momentum of SGHMC
    mh_step: int
        the number of backward imputation steps
    sigma_list: list of floats
        gaussian noise for each layer of the network
    prior_sigma_0, prior_sigma_1: float
        variances for mixture gaussian prior
    lambda_n: float
        proportion for components of mixture gaussian prior
    para_lr_decay, impute_lr_decay: float
        decay factor for para_lr and impute_lr, respectively
    scalar_y: float
        when the output is standardized, the losses need to be converted back to the original scale by multiplying
        scalar_y, which is essentially the variance of the train set of y
    outcome_cat: bool
        the type of outcome variable
        if TRUE, the outcome variable is a categorical variable, and this is a regression task
        if False, the outcome variable is a numerical variable, and this is a classification task
    temperature:
        temperature scaling for treatment model
    CE_weight: None or torch tensor:
        when the outcome variable is categorical, the weight assigned to each class.
        can be used to deal with unbalanced dataset.


    output:
    para_path: dictionary
            parameter values for each epoch
    para_grad_path: dictionary
            parameter gradient values for each epoch
    para_gamma_path: dictionary
            indicator of connection selection for each epoch
    input_gamma_path: dictionary
            var_selected_out: indicator for variable selection for outcome variable
            num_selected_out: number of selected input variables for outcome variable
            var_selected_treat: indicator for variable selection for treatment
            num_selected_treat: number of selected input variables for treatment
    performance: dictionary
            model performance for each epoch
    impute_lrs: list of floats
            starting value of impute_lrs for refining network weight
    likelihoods: list of floats
            stores the likelihoods for each hidden layer after final updates of the neural network
            the likelihoods are used to calculate BIC (note: the likelihoods are calculated based on the standardized
            dataset)
    """

    # save training and validation performance
    out_train_loss_path = []
    out_val_loss_path = []
    treat_train_loss_path = []
    treat_val_loss_path = []
    treat_train_acc_path = []
    treat_val_acc_path = []
    performance = dict(out_train_loss=out_train_loss_path, out_val_loss=out_val_loss_path,
                       treat_train_loss=treat_train_loss_path, treat_val_loss=treat_val_loss_path,
                       treat_train_acc=treat_train_acc_path, treat_val_acc=treat_val_acc_path)
    if outcome_cat:
        out_train_acc_path = []
        out_val_acc_path = []
        performance.update([('out_train_acc', out_train_acc_path), ('out_val_acc', out_val_acc_path)])

    para_path = {}
    para_grad_path = {}
    para_gamma_path = {}

    # save hidden likelihoods for calculating BIC
    hidden_likelihood = np.zeros(net.num_hidden+1)

    # initial value of decaying impute_lrs and para_lrs
    step_impute_lrs = impute_lrs.copy()
    init_para_lrs = []
    for i in range(net.num_hidden+1):
        init_para_lrs.append(optimizer_list[i].param_groups[0]['lr'])

    if mode == "train":
        # save parameter values, indicator variables, and selected input variables for each epoch
        input_gamma_path = dict(var_selected_out={}, var_selected_treat={}, num_selected_out=[],
                                num_selected_treat=[])

        # # para_lr decay
        # scheduler_list = []
        # for j in range(net.num_hidden + 1):
        #     # decay para_lr once val_loss is not decreasing
        #     scheduler_list.append(ReduceLROnPlateau(optimizer_list[j], patience=20, factor=0.95, eps=1e-14))

    # settings for output loss functions
    if outcome_cat:
        # calculate the weight for each class

        out_loss = nn.CrossEntropyLoss(weight=CE_weight)
        out_loss_sum = nn.CrossEntropyLoss(weight=CE_weight, reduction='sum')
    else:
        out_loss = nn.MSELoss()
        out_loss_sum = nn.MSELoss(reduction='sum')

    # treatment loss function
    if isinstance(net.treat_node, (list, tuple, np.ndarray)):
        treat_loss = nn.CrossEntropyLoss()
    else:
        treat_loss = nn.BCELoss()

    # intermediate values for prior gradient calculation
    c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
    c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1

    # threshold for sparsity
    threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
            0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

    # # average time for each epoch
    # accumulated_time = 0

    # training
    for epoch in range(epochs):
        print("Epoch" + str(epoch))

        # tic = time.time()

        if mode == "train":
            # impute_lr decay and para_lr decay
            for i in range(net.num_hidden):
                step_impute_lrs[i] = impute_lrs[i]/(1+impute_lrs[i]*epoch**impute_lr_decay)
                optimizer_list[i].param_groups[0]['lr'] = init_para_lrs[i]/(1+init_para_lrs[i]*epoch**para_lr_decay)
            optimizer_list[-1].param_groups[0]['lr'] = init_para_lrs[-1]/(1+init_para_lrs[-1]*epoch**para_lr_decay)

        # print("impute_lrs", step_impute_lrs)
        # for i in range(net.num_hidden):
        #     print("para_lr", optimizer_list[i].param_groups[0]['lr'])

        for y, treat, x in train_data:
            # backward imputation
            hidden_list = net.backward_imputation(mh_step, step_impute_lrs, alpha, out_loss_sum, sigma_list, x,
                                                  treat, y, temperature)

            # parameter update
            for para in net.parameters():
                para.grad = None

            with torch.no_grad():  # prior gradient, update for each parameter
                for para in net.parameters():
                    temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                    prior_grad = temp.div(len(train_data)*batch_size)
                    para.grad = prior_grad

            for layer_index in range(net.num_hidden + 1):
                forward_hidden = net.module_list[0](x)
                likelihood = net.likelihood(forward_hidden, hidden_list, layer_index, out_loss_sum, sigma_list, y,
                                            temperature) / batch_size
                optimizer = optimizer_list[layer_index]
                likelihood.backward()
                # gradient clipping on hidden layers
                if layer_index < net.num_hidden + 1:
                    torch.nn.utils.clip_grad_norm_(net.module_list[layer_index].parameters(),
                                               max_norm=1/(2*sigma_list[layer_index]), norm_type=2)
                optimizer.step()

                if epoch == epochs-1:
                    with torch.no_grad():
                        # need to recalculate likelihood afater the last parameter update
                        # make sure that treat loss have the same weight as outcome loss
                        forward_hidden = net.module_list[0](x)
                        likelihood = net.likelihood(forward_hidden, hidden_list, layer_index, out_loss_sum, sigma_list, y,
                                                    temperature)
                        hidden_likelihood[layer_index] += likelihood

        # calculate training performance
        out_train_loss, out_train_correct, treat_train_loss, treat_train_correct = 0, 0, 0, 0
        with torch.no_grad():
            for y, treat, x in train_data:
                pred, ps = net.forward(x, treat, temperature)
                out_train_loss += out_loss(pred, y).item()
                treat_train_loss += treat_loss(ps, treat).item()  # note that for BCELoss the input has to be probability
                if isinstance(net.treat_node, (list, tuple, np.ndarray)):
                    treat_train_correct += (ps.argmax(dim=1) == treat.argmax(dim=1)).sum().item()
                else:
                    treat_train_correct += ((ps > 0.5) == treat).sum().item()
                if outcome_cat:
                    out_train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if outcome_cat is False:
            # use RMSE as the model performance metric for regression tasks
            out_train_loss = np.sqrt(out_train_loss/len(train_data)) * scalar_y
        else:
            out_train_loss /= len(train_data)
        out_train_loss_path.append(out_train_loss)
        print(f"Avg train loss: {out_train_loss:>8f} \n")
        if outcome_cat:
            out_train_correct /= len(train_data.dataset)
            out_train_acc_path.append(out_train_correct)
            print(f"train accuracy: {out_train_correct:>8f} \n")

        treat_train_loss /= len(train_data)
        treat_train_loss_path.append(treat_train_loss)
        treat_train_correct /= len(train_data.dataset)
        treat_train_acc_path.append(treat_train_correct)

        # calculate validation performance
        out_val_loss, out_val_correct, treat_val_loss, treat_val_correct = 0, 0, 0, 0
        with torch.no_grad():
            for y, treat, x in val_data:
                pred, ps = net.forward(x, treat, temperature)
                out_val_loss += out_loss(pred, y).item()
                treat_val_loss += treat_loss(ps, treat).item()
                if isinstance(net.treat_node, (list, tuple, np.ndarray)):
                    treat_val_correct += (ps.argmax(dim=1) == treat.argmax(dim=1)).sum().item()
                else:
                    treat_val_correct += ((ps > 0.5) == treat).sum().item()
                if outcome_cat:
                    out_val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if outcome_cat is False:
            # use RMSE as the model performance metric for regression tasks
            out_val_loss = np.sqrt(out_val_loss/len(val_data)) * scalar_y
        else:
            out_val_loss /= len(val_data)
        out_val_loss_path.append(out_val_loss)
        print(f"Avg val loss: {out_val_loss:>8f} \n")
        if outcome_cat:
            out_val_correct /= len(val_data.dataset)
            out_val_acc_path.append(out_val_correct)
            print(f"val accuracy: {out_val_correct:>8f} \n")

        treat_val_loss /= len(val_data)
        treat_val_loss_path.append(treat_val_loss)
        treat_val_correct /= len(val_data.dataset)
        treat_val_acc_path.append(treat_val_correct)

        # toc = time.time()
        # accumulated_time += toc - tic

        # save parameter values and selected connections
        para_path_temp = {str(epoch): {}}
        para_grad_path_temp = {str(epoch): {}}
        para_gamma_path_temp = {str(epoch): {}}

        for name, para in net.named_parameters():
            para_path_temp[str(epoch)][name] = torch.clone(para).data.cpu().numpy().tolist()
            para_grad_path_temp[str(epoch)][name] = torch.clone(para.grad).data.cpu().numpy().tolist()
            para_gamma_path_temp[str(epoch)][name] = (para.abs() > threshold).data.cpu().numpy().tolist()
        para_path.update(para_path_temp)
        para_grad_path.update(para_grad_path_temp)
        para_gamma_path.update(para_gamma_path_temp)

        if mode == "train":
            # select input variable
            var_ind_out = np.identity(len(para_gamma_path[str(epoch)]['0.weight'][0]), dtype=bool)
            for i, (name, para) in enumerate(net.named_parameters()):
                if i % 2 == 0:
                    var_ind_out = np.matmul(para_gamma_path[str(epoch)][name], var_ind_out)
                    if i/2 == net.treat_layer:
                        var_ind_treat = np.copy(var_ind_out[net.treat_node, :])
                        var_ind_out[net.treat_node, :] = np.zeros_like(var_ind_out[net.treat_node, :])
            var_ind_out = np.max(var_ind_out, 0)
            if isinstance(net.treat_node, (list, tuple, np.ndarray)):
                var_ind_treat = np.prod(var_ind_treat, axis=0)
            num_selected_out = np.sum(var_ind_out)
            num_selected_treat = np.sum(var_ind_treat)
            input_gamma_path['var_selected_out'][str(epoch)] = var_ind_out.tolist()
            input_gamma_path['var_selected_treat'][str(epoch)] = var_ind_treat.tolist()
            input_gamma_path['num_selected_out'].append(num_selected_out.astype("float64"))
            input_gamma_path['num_selected_treat'].append(num_selected_treat.astype("float64"))
            print('number of selected input variable for outcome:', num_selected_out)
            print('number of selected input variable for treatment:', num_selected_treat)

            # # para_lr decay
            # for layer_index in range(net.num_hidden + 1):
            #     scheduler = scheduler_list[layer_index]
            #     # scheduler.step()
            #     # print("para_lr", scheduler.get_last_lr())
            #     scheduler.step(val_loss)
            #     # print("para_lr", optimizer_list[layer_index].param_groups[0]['lr'])

    # print("average time per epoch", accumulated_time/epochs)

    if mode == "pretrain":
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      performance=performance)
    else:
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      input_gamma_path=input_gamma_path, performance=performance,
                      impute_lrs=step_impute_lrs, likelihoods=hidden_likelihood)

    return output
