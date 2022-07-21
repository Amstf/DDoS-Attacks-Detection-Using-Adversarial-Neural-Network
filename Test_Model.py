import pandas as pd
import numpy as np
import tensorflow as tf
# data split

from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve , auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns  
from keras.callbacks import ModelCheckpoint
# from IDSTraining import data_split
from LSTMforReal import *
def FF(df,features):

    list_1=list(features["col_name"][:20])
    list_1.append('Label')
    ds=pd.DataFrame()
    ds=df[list_1]
    return ds
# global parameter 
data_path="combined.csv"
dataset=pd.read_csv(data_path)

model_path=("wandb/Real_fake/files/model-best.h5")
print(len(list(dataset.columns)))
print(dataset.head(5))
n_features=len(list(dataset.drop(columns=["Label"]).columns))

def pre_processing(X_train , X_test, y_train , y_test):

    scaler= MinMaxScaler()
    le = LabelEncoder()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    print("intances per label in training set \n",y_train.value_counts())
    y_train=le.fit_transform(y_train)

    print("intances per label in test set \n",y_test.value_counts())
    y_test=le.fit_transform(y_test)

    print(X_train.shape)
    print(X_test.shape)

    y_train = np.asarray(y_train).astype("float32").reshape((-1,1))
    y_test = np.asarray(y_test).astype("float32").reshape((-1,1))

    # reshape input data to LSTM format [samples , time_steps, features]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    print(f"shape of X_train:", X_train.shape)
    print(f"shape of X_test:", X_test.shape)

    return X_train , X_test, y_train , y_test


def load_model(model_path):
    # restore the model file "model.h5" from a specific run by user "alimustapha"
    # in project "save_and_restore" from run "23yow4re"

    # use the "name" attribute of the returned object if your framework expects a filename, e.g. as in Keras
    model = tf.keras.models.load_model(model_path)

    # Check its architecture
    print(model.summary())
    model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam',metrics=['accuracy'])
    print("**************************MODEL COMPILED**************************")

    return model
def plot_cm(y_true, y_pred, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    plt.savefig("Real_Fake_LSTM.png")
    print(classification_report(y_true, y_pred))
    print(cm)



# Label_mean = dataset.Label.mean()
# dataset['Label'] = dataset['Label'] > Label_mean
# dataset["Label"] = dataset["Label"].astype(int)
fe=pd.read_csv("DDoS_Functional_Features.csv")

data_path="combined.csv"
real=get_real_data(data_path)
real=drop_d(real,fe)
fake=get_fake_data("Results/generatedData.csv")
dataset=get_combined_data(real,fake)

X_train , X_test, y_train , y_test = data_split(dataset)
X_train , X_test, y_train , y_test = pre_processing(X_train , X_test, y_train , y_test)
# loading model
model = load_model(model_path)
print("model Loaded")


# predicting on test set
print("making pridtction")
y_test_pred_prob = model.predict(X_train, verbose=1)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

plot_cm(y_train, y_test_pred)
