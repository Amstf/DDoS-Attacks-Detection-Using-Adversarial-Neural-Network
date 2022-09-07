import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns  
from plotly import express as px
from plotly import graph_objects as go
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score


# from IDSTraining import data_split

from utils import *

def load_model(model_path):
    # restore the model file "model.h5" from a specific run by user "alimustapha"
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




def test_model(model_path,poly_df):
  X=poly_df.iloc[:, :-1 ].to_numpy()
  y = poly_df.iloc[:,-1].to_numpy()
  labelencoder= LabelEncoder()
  y = labelencoder.fit_transform(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)

  scale_X= MinMaxScaler()
  X_train= scale_X.fit_transform(X_train)
  X_test= scale_X.transform(X_test)
  print("after spliting the data:\n")
  print("training data length:", len(X_train))
  print("test data length:", len(X_test))


  ### reshape input data to LSTM format [samples, time_steps, features]
  X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
  X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
  print(f"shape of X_train:", X_train_lstm.shape)
  print(f"shape of X_test:", X_test_lstm.shape)
  model = tf.keras.models.load_model(model_path)

  # Check its architecture
  model.summary()
  model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam',metrics=['accuracy'])
  # predicting on training set
  y_test_pred_prob = model.predict(X_test_lstm)
  y_test_pred = np.argmax(y_test_pred_prob, axis=1)
  return y_test,y_test_pred
  

def modification(nb_features):
  concate_path='/content/drive/MyDrive/ParisTech/new/combined.csv'
  ddos=pd.read_csv('/content/drive/MyDrive/ParisTech/models/LSTM/genm.csv')
  df = pd.read_csv(concate_path)
  normal=df.loc[df['Label'] == 0]

  ddos['Label'] = 1
  ddos=ddos.drop([' Fwd Header Length.1'],axis=1)
  list_1=list(fe.col_name[0:nb_features])
  normal=normal[:236494]
  for i in list_1:
    print(i)
    a=ddos.columns.get_loc(i)
    ddos=ddos.drop(columns=[i],axis=1)
    ddos.insert(a,i,list(normal[i]))
  poly_df = pd.concat([normal[:236494], ddos])
  return poly_df


model_path='/content/model-best.h5'


  
result_df=pd.DataFrame()
result_df.insert(0,"y_Test" , y_test)
modifications=[0,8,16,20]




data_path="combined.csv"
dataset=pd.read_csv(data_path)

model_path=("model-best.h5")
print(len(list(dataset.columns)))
print(dataset.head(5))
n_features=len(list(dataset.drop(columns=["Label"]).columns))
for i in modifications:
  df=modification(i)
  y_test,y_pred,y_test_pred_prob=test_model(model_path,df)
  result_df.insert(1,"Y_{}".format(i), y_test_pred_prob[:, 1 ])

aur_roc_data = []

for i in modifications:
  fprs, tprs, thresholds = roc_curve(y, y_test_pred_prob)
  score = roc_auc_score(y, y_pred)
  aur_roc_data.append(pd.DataFrame(zip([i]*len(fprs), fprs, tprs, thresholds, [score]*len(fprs)), columns = ["modification", "fpr", "tpr", "threshold", "score"]))
aur_roc_data = pd.concat(aur_roc_data)

fig = px.line(aur_roc_data, x="fpr", y="tpr", color="modification",
                 labels= {"fpr": "False Positive Rate", "tpr": "True Positive Rate"},
                 title = "AUC-ROC Curve",
                 width=600, height=600,
                 hover_data= {"score":True}
                 )
fig.update_layout(plot_bgcolor = "skyblue")

fig.for_each_trace(lambda a: a.update(name=a.name.split("=")[-1]))
fig.add_trace(
    go.Scatter(x= np.arange(11)/10,
               y=np.arange(11)/10,
               showlegend=False,
           
               marker=dict(color="Black", size=1),
               )
    )
fig.save(figure_path)
