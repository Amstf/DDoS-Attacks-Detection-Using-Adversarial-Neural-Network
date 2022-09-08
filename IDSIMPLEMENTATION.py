# import libraries
import pandas as pd
import numpy as np
# data split

from sklearn.model_selection import train_test_split
from collections import Counter

# data preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
from utils import *


# model building
import tensorflow as tf




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
        
        


def load_data(real_path,fake):
    real=pd.read_csv(real_path)
    
    normal=real.loc[real['Label'] == 0]
    DDoS=real.loc[real['Label'] == 1]
    normal=normal[:4000]
    DDoS=DDoS[:4000]
    real = pd.concat([normal, DDoS])
    fake['Label']=0
    real['outcome']=0
    fake['outcome']=1
    df = pd.concat([real, fake])
    return df

def load_model(model_path):
    model= tf.keras.models.load_model(model_path)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam',metrics=['accuracy'])
    return model

def modification(nb_features):
  ddos=pd.read_csv('Results/generatedAllData.csv')
  normal=pd.read_csv('Results/generatedAllData.csv')
  fe=pd.read_csv("DDoS_Functional_Features.csv")
  ddos['Label'] = 1
  list_1=list(fe.col_name[0:nb_features])
  for i in list_1:
    print(i)
    a=ddos.columns.get_loc(i)
    ddos=ddos.drop(columns=[i],axis=1)
    ddos.insert(a,i,list(normal[i]))
  return ddos


def data_split(df,IDS=True):
    if IDS:
        y=df.Label
        X=df.drop(columns=["Label"])
    else :
        y=df.outcome
        X=df.drop(columns=["Label","outcome"])    
    

    # X=df[columns]
    labels=y.unique()
    classes=y.nunique()
    print(X.shape)
    print("number of Label", classes)
    print("instances per label\n", y.value_counts())
    print("label",labels)
    
    # split the dataset into 80% for training and 20% for testing
    # X_train , X_test, y_train , y_test = train_test_split(X,y, random_state=42 , stratify=y, shuffle=True,test_size=0.2)





    return X , y 

def pre_processing(X_test, y_test):
    scaler= MinMaxScaler()
    le = LabelEncoder()
    X_test=scaler.fit_transform(X_test)
    # print("intances per label in test set \n",y_test.value_counts())
    y_test=le.fit_transform(y_test)
    print(X_test.shape)
    y_test = np.asarray(y_test).astype("float32").reshape((-1,1))
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    return  X_test , y_test


def test_model(df,X_test,model):
  # predicting on training set
  y_pred_prob = model.predict(X_test)
  y_test_pred = np.argmax(y_pred_prob, axis=1)
  
  df['preds']=y_test_pred
  return y_pred_prob,df

def drop_d(df,features):
    list_1=list(features["col_name"][:20])
    for i in(list_1):
        df=df.drop(i,axis=1)
    return df
def plot_cm(y_true, y_pred_prob,name, figsize=(10,10)):
    y_pred = np.argmax(y_pred_prob, axis=1)
    
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
    plt.savefig(name)
    print(classification_report(y_true, y_pred))
    print(cm)
def FF(df,features):

    list_1=list(features["col_name"][:20])
    list_1.append('Label')
    ds=pd.DataFrame()
    ds=df[list_1]
    return ds
model=load_model("wandb/OnlyFF/files/model-best.h5")
real_fake=load_model("wandb/REAL/files/model-best.h5")
fe=pd.read_csv("/home/infres/amustapha/DDoS/GAN/DDoS_Functional_Features.csv")

real_data="combined.csv"
fake=modification(16)

df=load_data(real_data,fake)
data=drop_d(df,fe)
X_test , y_test =data_split(data,False)
X_test,  y_test=pre_processing( X_test,y_test)
y_pred_prob,result=test_model(df,X_test,real_fake)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_prob[:,1])
auc_keras = auc(fpr_keras, tpr_keras)
print("Real/fake auc",auc_keras)
# plot_cm(y_test, y_pred_prob,"REALorFAKE.png")
y_pred = np.argmax(y_pred_prob, axis=1)

print(confusion_matrix(y_test, y_pred, labels=np.unique(y_test)))





result_df = result.loc[result['preds'] == 0]
result_df=result_df.drop('preds',axis=1)
result_df=FF(result_df,fe)
X_test , y_test =data_split(result_df,True)
X_test,  y_test=pre_processing( X_test,y_test)
y_pred_prob,result=test_model(result_df,X_test,model)
# plot_cm(y_test, y_pred_prob,"ddosorNormal.png")
# plot_cm(y_test, y_pred_prob,"REALorFAKE.png")
y_pred = np.argmax(y_pred_prob, axis=1)

print(confusion_matrix(y_test, y_pred, labels=np.unique(y_test)))
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_prob[:,1])
auc_keras = auc(fpr_keras, tpr_keras)
print(auc_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='AUC (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
plt.savefig('ROC.png',bbox_inches = 'tight')







