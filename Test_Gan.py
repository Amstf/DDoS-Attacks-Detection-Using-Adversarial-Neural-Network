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
from IDSTraining import data_split
from utils import *

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



def drop_ff(df,features):
    list1=list(features["col_name"][:20])
    for i in (list1):
        df=df.drop(i,axis=1)
    n_features=len(list(df.drop("Label",axis=1).columns))
    print(n_features)
    print(df.head())
    return df,n_features
model_path=("wandb/onlyFF/files/model-best.h5")

def combine(dataset):
    normal=dataset.loc[dataset["Label"]==0]
    generated=pd.read_csv("Results/generatedAllData.csv")
    generated["Label"]=1
    
    # concatenating df1 and df2 along rows
    df = pd.concat([generated, normal], axis=0)
    return df
    
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
    plt.savefig("confusion_matrix_AllFF.png")
    print(classification_report(y_true, y_pred))
    print(cm)





# global parameter 
data_path="combined.csv"
dataset=pd.read_csv(data_path)
fe=pd.read_csv("DDoS_Functional_Features.csv")
# dataset,n_features=drop_ff(dataset,fe)
# print(n_features)
dataset=combine(dataset)


            
# Label_mean = dataset.Label.mean()
# dataset['Label'] = dataset['Label'] > Label_mean
# dataset["Label"] = dataset["Label"].astype(int)

# y=dataset.Label
# x=dataset.drop(columns=  ["Label"])

# X_train , X_test, y_train , y_test = train_test_split(x,y, random_state=42 , stratify=y, shuffle=True,test_size=0.8)

# X_test = X_test.values #returns a numpy array

# X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])    
# y_test = np.asarray(y_test).astype("float32").reshape((-1,1))

# # loading model
# model = load_model(model_path)
# print("model Loaded")

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


X_train , X_test, y_train , y_test = data_split(dataset)
X_train, X_test, y_train, y_test = pre_processing(X_train , X_test, y_train , y_test)

# loading model
model = load_model("wandb/AllF/files/model-best.h5")
print("model trained")


# predicting on test set
print("making pridtction")
y_test_pred_prob = model.predict(X_train, verbose=1)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

plot_cm(y_train, y_test_pred)
