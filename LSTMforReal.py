# import libraries
import pandas as pd
import numpy as np
# data split

from sklearn.model_selection import train_test_split
from collections import Counter

# data preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import os


# model building
import tensorflow as tf
from tensorflow.keras import Model , Sequential,Input, backend
from tensorflow.keras.layers import LSTM , Dense , Dropout , Flatten
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from utils import *


import wandb
from wandb.keras import WandbCallback

gpus=tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus=tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus),"Phusical GPUs", len(logical_gpus),"Logical GPUS")
    except RuntimeError as e:
        print(e)





def build_LSTM_model(n_features,n_classes):

    model=Sequential()

    model.add(Input(shape=(None, n_features),name="input"))

    model.add(LSTM(units=30,name="LSTM_layer"))
    model.add(Dense(256, activation = 'relu', name="Dense_Layer"))
    model.add(Dropout(0.5,name="Dropout_Layer"))
    model.add(Dense(n_classes, activation="sigmoid", name="Output"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam',metrics=['accuracy'])
    print(model.summary())
    print("NUMBER OF FEATURES :",n_features)
    return model


def train_model(model, X_train , y_train,epochs=30,batch_size=32,validation_split=0.2 ):
    
    callback = EarlyStopping(patience=20, mode='min', restore_best_weights=True)
    backend.clear_session()
    history = model.fit(X_train,y_train, 
                        epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[callback,WandbCallback()])
    # "model.h5" is saved in wandb.run.dir & will be uploaded at the end of training
    model.save(os.path.join(wandb.run.dir, "model.h5"))

# fe=pd.read_csv("/home/infres/amustapha/DDoS/GAN/DDoS_Functional_Features.csv")
    
# fe=pd.read_csv("DDoS_Functional_Features.csv")

# data_path="combined.csv"
# real=get_real_data(data_path)
# fake=get_fake_data("Results/generatedAllclasses.csv")

# real=drop_FF(real,fe)

# fake=drop_FF(fake,fe)

# dataset=get_combined_data(real,fake)

# X_train , X_test, y_train , y_test = data_split(dataset)
# X_train , X_test, y_train , y_test = pre_processing(X_train , X_test, y_train , y_test)

# model = build_LSTM_model(38, 2)
# wandb_login()
# train_model(model, X_train , y_train)

    