
from sklearn.model_selection import train_test_split
from collections import Counter
import os

import tensorflow as tf
from tensorflow.keras import  Sequential,Input, backend
from tensorflow.keras.layers import LSTM , Dense , Dropout 
from tensorflow.keras.callbacks import EarlyStopping


import wandb
from wandb.keras import WandbCallback
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

def wandb_login():
    wandb.login()
    wandb.init(project="without_features", config={"hyper":"paramet"})

    


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


def train_model(model, X_train , y_train ):
    
    callback = EarlyStopping(patience=20, mode='min', restore_best_weights=True)
    backend.clear_session()
    history = model.fit(X_train,y_train, 
                        epochs=30, batch_size=32, validation_split=0.2, callbacks=[callback,WandbCallback()])
    # "model.h5" is saved in wandb.run.dir & will be uploaded at the end of training
    model.save(os.path.join(wandb.run.dir, "model.h5"))

# fe=pd.read_csv("/home/infres/amustapha/DDoS/GAN/DDoS_Functional_Features.csv")

# data_path="combined.csv"
# dataset= read_data(data_path)
# dataset=drop_d(dataset,fe)
# a=len(dataset.columns)
# X_train , X_test, y_train , y_test = data_split(dataset)
# X_train , X_test, y_train , y_test = pre_processing(X_train , X_test, y_train , y_test)
# del(dataset)
# model = build_LSTM_model(38, 2)
# wandb_login()
# train_model(model, X_train , y_train)

    