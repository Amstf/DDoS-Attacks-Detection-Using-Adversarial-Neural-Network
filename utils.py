# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler


def drop_FF(df,features):
    list_1=list(features["col_name"][:20])
    for i in(list_1):
        df=df.drop(i,axis=1)
    return df

def get_real_data(data_path):
    df=pd.read_csv(data_path)
    df=df.loc[df["Label"]==1]
    df=df.iloc[:8000]
    df.drop("Label",axis=1)
    df["Label"]=0
    return df

def get_fake_data(data_path):
    df=pd.read_csv(data_path)
    df["Label"]=1
    return df

def get_combined_data(real_df,fake_df):
    return pd.concat([real_df, fake_df], axis=0)
def data_split(df):
    
    y=df.Label
    X=df.drop(columns=["Label"])
    # X=df[columns]
    
    labels=y.unique()
    classes=y.nunique()


    print(X.shape)
    print("number of Label", classes)
    print("instances per label\n", y.value_counts())
    print("label",labels)
    
    # split the dataset into 80% for training and 20% for testing
    X_train , X_test, y_train , y_test = train_test_split(X,y, random_state=42 , stratify=y, shuffle=True,test_size=0.2)


    print("after spliting the data :\n")
    print("training data length:", len(X_train))
    print("test data length:", len(X_test))

    return X_train , X_test, y_train , y_test

def pre_processing(X_train , X_test, y_train , y_test):

    scaler= MinMaxScaler()
    le = LabelEncoder()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    # print("intances per label in training set \n",y_train.value_counts())
    y_train=le.fit_transform(y_train)

    # print("intances per label in test set \n",y_test.value_counts())
    y_test=le.fit_transform(y_test)

    print(X_train.shape)
    print(X_test.shape)

    y_train = np.asarray(y_train).astype("float32").reshape((-1,1))
    y_test = np.asarray(y_test).astype("float32").reshape((-1,1))
    

    # reshape input data to LSTM format [samples , time_steps, features]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
   
    # print(f"shape of X_train:", X_train.shape)
    # print(f"shape of X_test:", X_test.shape)

    return X_train , X_test, y_train , y_test
