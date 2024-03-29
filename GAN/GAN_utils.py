import torch
import pandas as pd
import numpy as np
import torch
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def drop_function(df,features):
    list_1=list(features["col_name"][:20])
    for i in(list_1):
        df=df.drop(i,axis=1)
    n_features=len(list(df.columns))
    return df,n_features

def get_ohe_data(df):
    
    df_int = df.select_dtypes(['float', 'integer']).values
    continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
    ##############################################################
    scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
    df_int = scaler.fit_transform(df_int)

    df_cat = df.select_dtypes('object') 
    df_cat_names = list(df.select_dtypes('object').columns)
    numerical_array = df_int
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))


    final_array = np.hstack((numerical_array, ohe_array.toarray()))
    return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array


def get_original_data(df_transformed, df_orig, ohe, scaler):
    # df_int = df_orig.select_dtypes(['float','integer'])
    df_ohe_int = df_transformed[:, :df_orig.select_dtypes(['float', 'integer']).shape[1]]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
  
    # df_income = df_transformed[:,-1]
    # df_ohe_cats = np.hstack((df_ohe_cats, df_income.reshape(-1,1)))
    df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
    return df_int


def prepare_data(df, batch_size):
    #df = pd.concat([df_train, df_test], axis=0)

    ohe, scaler, discrete_columns, continuous_columns, df_transformed = get_ohe_data(df)


    input_dim = df_transformed.shape[1]

    #from sklearn.model_selection import train_test_split
    #################
    X_train, X_test = train_test_split(df_transformed,test_size=0.1, shuffle=True) #random_state=10)
    #X_train = df_transformed[:df_train.shape[0],:]
    #X_test = df_transformed[df_train.shape[0]:,:]

    data_train = X_train.copy()
    data_test = X_test.copy()

    data = torch.from_numpy(data_train).float()


    train_ds = TensorDataset(data)
    train_dl = DataLoader(train_ds, batch_size = batch_size, drop_last=True)
    return ohe, scaler, input_dim, discrete_columns, continuous_columns ,train_dl, data_train, data_test

def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_data)

    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def get_gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp

    return crit_loss
