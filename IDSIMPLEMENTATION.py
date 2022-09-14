# import libraries
from tabnanny import verbose
import pandas as pd
import numpy as np
# data split
from sklearn.model_selection import train_test_split
from collections import Counter
# data preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import os
# model building
import tensorflow as tf
from utils import drop_FF, data_split,pre_processing,load_models


gpus=tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        #currentlu, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus=tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus),"Phusical GPUs", len(logical_gpus),"Logical GPUS")
    except RuntimeError as e:
        #,e,pry growth must be set before GPUs have been initialized
        print(e)
        




def get_real_data(data_path):
    df=pd.read_csv(data_path)
    df=df.iloc[:8000]
    ds=df.drop("Label",axis=1)
    ds["Label"]=0
    return df,ds

def get_fake_data(data_path):
    df=pd.read_csv(data_path)
    df["Label"]=1
    return df

def get_combined_data(real_df,fake_df):
    return pd.concat([real_df, fake_df], axis=0)


def modification(nb_features,fe,df):
  concate_path='/content/drive/MyDrive/ParisTech/new/combined.csv'
  ddos=pd.read_csv('/content/drive/MyDrive/ParisTech/models/LSTM/genm.csv')
  df = pd.read_csv(concate_path)
  normal=df.loc[df['Label'] == 0]

  ddos['Label'] = 1
  ddos=ddos.drop([' Fwd Header Length.1'],axis=1)
  list_1=list(fe.col_name[0:nb_features])
  normal=normal[:8000]
  for i in list_1:
    print(i)
    a=ddos.columns.get_loc(i)
    ddos=ddos.drop(columns=[i],axis=1)
    ddos.insert(a,i,list(normal[i]))
  poly_df = pd.concat([normal[:8000], ddos])
  return poly_df
def pridect_Real(model,df):
    ds=[]
    for i in (df):
        model.predict(i)
        i=np.argmax(i,axis=0)
        if(i==0):
            ds.append(i,verbose=1)
    return ds
            
    
fe=pd.read_csv("/home/infres/amustapha/DDoS/GAN/DDoS_Functional_Features.csv")

real_path="combined.csv"
fake_path="Results/generatedAllData.csv"
Df,real=get_real_data(real_path)
fake=get_fake_data(fake_path)
combined_data=get_combined_data(real,fake)
Critic_data=drop_FF(combined_data,fe)
X_train , X_test, y_train , y_test = data_split(Critic_data)
X_train , X_test, y_train , y_test = pre_processing(X_train , X_test, y_train , y_test)
Criti=load_models("wandb/Real_fake/files/model-best.h5")
ds=pridect_Real(X_train)



pridect_Real(Criti,Critic_data)

# IDS=load_models("wandb/OnlyFF/files/model-best.h5")
