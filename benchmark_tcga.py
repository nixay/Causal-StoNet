from benchmark_data import *
from causalml.inference.meta import BaseDRLearner, BaseRLearner, BaseXLearner
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import json
import pickle
import numpy as np
from sklearn.model_selection import KFold
import argparse

parser = argparse.ArgumentParser(description='Run benchmark for 401k data')
parser.add_argument('--seed', default=1, type=int, help='set seed')

args = parser.parse_args()

model_seed = args.seed
np.random.seed(model_seed)
# network configure
common_args = dict(activation='tanh', solver='sgd', alpha=0, batch_size=200, learning_rate='invscaling', max_iter=10000,
                   early_stopping=True, random_state=model_seed)
treatment_model = MLPClassifier(hidden_layer_sizes=(128,), **common_args)
outcome_model = MLPClassifier(hidden_layer_sizes=(128, 32, 8), **common_args)

# load data
y, treat, x = TCGA_bench()

ates_dr, ates_r, ates_x = np.zeros((2,3)), np.zeros((2,3)), np.zeros((2,3))
std_dr, std_r, std_x = np.zeros(2), np.zeros(2), np.zeros(2)
cate_dr, cate_r, cate_x = [], [], []

# DR-Learner
learner_dr = BaseDRLearner(control_outcome_learner=outcome_model, treatment_outcome_learner=outcome_model,
                           treatment_effect_learner=LinearRegression(), control_name=2)
learner_dr.fit(x, treat, y)
cate_dr = learner_dr.predict(X=x, treatment=treat, y=y)
ate_dr = cate_dr.mean(axis=0)

# to get the standard deviation of averaged ATE for each cross validation
models_tau = learner_dr.models_tau
ates_dr[0, :] = np.r_[[model.predict(x) for model in models_tau[0]]].mean(axis=1)
ates_dr[1, :] = np.r_[[model.predict(x) for model in models_tau[1]]].mean(axis=1)
std_dr[0] = np.sqrt(ates_dr[0, :].var()/3)
std_dr[1] = np.sqrt(ates_dr[1, :].var()/3)

cv = KFold(n_splits=3, shuffle=True, random_state=1)
split_indices = [index for _, index in cv.split(y)]

for ifold in range(3):
    print('cross val', ifold)

    train_idx = np.concatenate((split_indices[ifold], split_indices[(ifold + 1) % 3]))
    val_idx = split_indices[(ifold + 2) % 3]

    y_train, y_val = y[train_idx], y[val_idx]
    treat_train, treat_val = treat[train_idx], treat[val_idx]
    x_train, x_val = x[train_idx], x[val_idx]

    # R-Learner
    learner_r = BaseRLearner(outcome_learner=outcome_model, effect_learner=LinearRegression(),
                             propensity_learner=treatment_model, control_name=2, n_fold=2, random_state=model_seed)
    learner_r.fit(x_train, treat_train, y_train)
    cate = learner_r.predict(x_val)
    cate_r.append(cate)
    ates_r[:, ifold] = cate.mean(axis=0)

    # X-Learner
    learner_x = BaseXLearner(control_outcome_learner=outcome_model, treatment_outcome_learner=outcome_model,
                             control_effect_learner=LinearRegression(), treatment_effect_learner=LinearRegression(),
                             control_name=2)
    learner_x.fit(x_train, treat_train, y_train)
    cate = learner_x.predict(x_val, treat_val, y_val)
    cate_x.append(cate)
    ates_x[:, ifold] = cate.mean(axis=0)

std_r[0] = np.sqrt(ates_r[0, :].var()/3)
std_r[1] = np.sqrt(ates_r[1, :].var()/3)

std_x[0] = np.sqrt(ates_x[0, :].var()/3)
std_x[1] = np.sqrt(ates_x[1, :].var()/3)

ate_r = ates_r.mean(axis=1)
ate_x = ates_x.mean(axis=1)

cate_r = np.concatenate(cate_r)
cate_x = np.concatenate(cate_x)

ate_result = dict(ate=dict(dr=ate_dr.tolist(), r=ate_r.tolist(), x=ate_x.tolist()),
                  sd=dict(dr=std_dr.tolist(), r=std_r.tolist(), x=std_x.tolist()))
cate_result = dict(dr=cate_dr, r=cate_r, x=cate_x)

# save the results to json file

ate_file = open('./benchmark/bench_tcga_ate.json', "w")
json.dump(ate_result, ate_file, indent="")
ate_file.close()

with open('./benchmark/bench_tcga_cate.pkl', 'wb') as f:
    pickle.dump(cate_result, f)
