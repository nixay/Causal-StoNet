from scipy.stats import truncnorm, bernoulli
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
import os


def SimData_bench(input_size, seed, data_size):
    one_count, zero_count = 0, 0  # count of the samples in treatment group and control group, respectively
    one_treat, one_x, one_y, one_y_count = ([] for _ in range(4))
    zero_treat, zero_x, zero_y, zero_y_count = ([] for _ in range(4))

    np.random.seed(seed)
    while min(one_count, zero_count) < data_size // 2:
        # generate x
        ee = truncnorm.rvs(-10, 10)
        x_temp = truncnorm.rvs(-10, 10, size=input_size) + ee
        x_temp /= np.sqrt(2)

        # nodes in the first hidden layer
        h11 = np.tanh(2*x_temp[0]+1*x_temp[3])
        h12 = np.tanh(-x_temp[0]-2*x_temp[4])
        h13 = np.tanh(2*x_temp[1]-2*x_temp[2])
        h14 = np.tanh(-2*x_temp[3]+1*x_temp[4])

        # nodes in the second hidden layer
        h21 = np.tanh(-2*h11+h13)
        h22 = h12-h13
        h23 = np.tanh(h13-2*h14)

        # generate treatment
        prob = np.exp(h22)/(1 + np.exp(h22))
        treat_temp = bernoulli.rvs(p=prob)

        # nodes in the third hidden layer
        h31 = np.tanh(1*h21-2*treat_temp)
        h32 = np.tanh(-1*treat_temp+2*h23)

        # counterfactual nodes in the third hidden layer
        h31_count = np.tanh(1*h21-2*(1-treat_temp))
        h32_count = np.tanh(-1*(1-treat_temp)+2*h23)

        # generate outcome variable
        y_temp = -4*h31+2*h32 + np.random.normal(0, 1)

        # generate counterfactual outcome variable
        y_count_temp = -4*h31_count+2*h32_count + np.random.normal(0, 1)

        if treat_temp == 1:
            one_count += 1
            one_x.append(x_temp)
            one_y.append(y_temp)
            one_treat.append(treat_temp)
            one_y_count.append(y_count_temp)
        else:
            zero_count += 1
            zero_x.append(x_temp)
            zero_y.append(y_temp)
            zero_treat.append(treat_temp)
            zero_y_count.append(y_count_temp)

    x = np.array(one_x[:(data_size // 2)] + zero_x[:(data_size // 2)])
    y = np.array(one_y[:(data_size // 2)] + zero_y[:(data_size // 2)])
    treat = np.array(one_treat[:(data_size // 2)] + zero_treat[:(data_size // 2)])
    y_count = np.array(one_y_count[:(data_size // 2)] + zero_y_count[:(data_size // 2)])

    return y, treat, x, y_count


def true_cate(y, treat, y_count):
    cate = (y - y_count) * (2*treat-1)
    return cate


def PensionData_bench():
    data = pd.read_csv("./raw_data/401k/401k.csv")
    cat_col = ['db', 'marr', 'male', 'twoearn', 'pira', 'nohs', 'hs', 'smcol', 'col', 'hown']

    y = np.array(data['net_tfa'], dtype=np.float32)
    treat = np.array(data['e401'], dtype=np.float32)
    cat_var = np.array(data[cat_col], dtype=np.float32)
    num_var = np.array(data.loc[:, ~data.columns.isin(['net_tfa', 'e401', *cat_col])], dtype=np.float32)

    # data preprocess
    x_scalar = RobustScaler()
    x_scalar.fit(num_var)
    num_var = np.array(x_scalar.transform(num_var))

    # concatenate preprocessed numerical variable and categorical variable
    x = np.concatenate((num_var, cat_var), axis=1)

    return y, treat, x


def ACIC_bench(dgp):
    csv_name = 'acic_homo' + str(dgp) + '.csv'
    csv_dir = os.path.join('./raw_data/acic', csv_name)
    data = pd.read_csv(csv_dir)

    # extract column names for categorical variables
    cat_col = []
    for col in data.columns:
        if data[col].abs().max() <= 10:
            if len(data[col].unique()) <= data[col].max() + 1:
                cat_col.append(col)
    cat_col = cat_col[1:]
    cat_var = np.array(data[cat_col], dtype=np.float32)
    num_var = np.array(data.loc[:, ~data.columns.isin(['Y', 'A', *cat_col])], dtype=np.float32)
    y = np.array(data['Y'], dtype=np.float32)
    treat = np.array(data['A'], dtype=np.float32)

    # data preprocess
    x_scalar = RobustScaler()
    x_scalar.fit(num_var)
    num_var = np.array(x_scalar.transform(num_var))

    # concatenate preprocessed numerical variable and categorical variable
    x = np.concatenate((num_var, cat_var), axis=1)

    return y, treat, x


def Twins_bench():
    data = pd.read_csv("./raw_data/twins/twins_data.csv")

    y = np.array(data['y'])
    treat = np.array(data['treat'], dtype=np.float32)
    x = np.array(data.loc[:, ~data.columns.isin(['y', 'treat', 'counter'])], dtype=np.float32)

    return y, treat, x


def Twins_bench_balanced(seed):
    data = pd.read_csv("./raw_data/twins/twins_data.csv")

    # undersampling the majority class
    d1 = data[(data.y==1) & (data.treat==1)]
    n1 = len(d1.index)
    d2 = data[(data.y==1) & (data.treat==0)].sample(n=int(n1*1.5), random_state=seed)
    d3 = data[(data.y==0) & (data.treat==1)].sample(n=int(n1*1.5), random_state=seed)
    d4 = data[(data.y==0) & (data.treat==0)].sample(n=int(n1*1.5), random_state=seed)
    d = pd.concat((d1, d2, d3, d4))

    y = np.array(d['y'])
    treat = np.array(d['treat'], dtype=np.float32)
    x = np.array(d.loc[:, ~data.columns.isin(['y', 'treat', 'counter'])], dtype=np.float32)

    return y, treat, x

def TCGA_bench():
    data = pd.read_csv("./raw_data/tcga/tcga_data_screened.csv")

    y = np.array(data['recur'])
    treat = np.argmax(np.array(data[['rad', 'sur', 'control']], dtype=np.float32), axis=1)
    x = np.array(data.loc[:, ~data.columns.isin(['recur', 'sur', 'rad', 'control'])],
                 dtype=np.float32)

    return y, treat, x
