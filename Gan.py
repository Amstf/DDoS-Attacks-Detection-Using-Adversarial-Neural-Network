import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import KernelCenterer
from sklearn.preprocessing import  MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout, Input ,LSTM
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from numpy.random import randn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from IDSTraining import read_data,pre_processing


data,n_feature=read_data("combined.csv")
print("Data Loaded")
features= data.columns[:-1]
print(features)

function_Features=pd.read_csv("/home/infres/amustapha/DDoS/GAN/DDoS_Functional_Features.csv")

def drop_function_Features(df,features):
    scaler= MinMaxScaler()
    # list_1=list(features["col_name"][:20])
    # for i in(list_1):
    #     df=df.drop(columns=i ,axis=1)
    df=df.loc[df['Label']==1]
    df=df.drop(columns=["Label"], axis=1)
    n_feature=len(list(df.columns))
    x = df.values #returns a numpy array
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)


    

    return df ,n_feature 
    



# define agenerate_latent_points to create random noise in the latent space 

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    # to reshape the random noise to the demensions for matching the input of generator model
    x_input = x_input.reshape(n_samples, latent_dim)
    
    return x_input

# function to generate or produce fake data. the input of the generator will be the created latent points or random noise

def generate_fake_samples(generator,latent_dim,n_samples):

    x_input=generate_latent_points(latent_dim,n_samples)
    # let the generator predict the input random noise and output a numpy array
    X=generator.predict(x_input)
    # because it is the fake data, the label will be 0
    y=np.zeros((n_samples,1))
    X = X.reshape(X.shape[0], 1, X.shape[1])
    y = np.asarray(y).astype("float32").reshape((-1,1))
    
    return X,y


# USED TO GENERATE REAL SAMPLES

def generate_real_samples(n):
    # it will randomly select  n samples from the real dataset
    X=data.sample(n)
    # the label for the real data sample is 
    y=np.ones((n, 1))
    # X=X.to_numpy()
    # X = X.reshape(X.shape[0], 1, X.shape[1])
    # y = np.asarray(y).astype("float32").reshape((-1,1))


    return X,y



# we will create a simple sequential model as generator.The kernel will be "he_uniform", the demension of the ouput layer is the same as the dimension of the dataset
def define_generator(latent_dim,n_outputs):

    model=Sequential()
    
    model.add(Dense(256, input_dim=latent_dim,activation="relu"))
    
    model.add(Dense(512,activation="relu"))
    
    model.add(Dense(1024,activation="relu"))
    
    model.add(Dense(32))
    model.add(Dense(n_outputs,activation="tanh"))
    
    return model


# we will definee the discriminator, the output layer is activated by @sigmoid@ function because it will discriminate the inmput smaples are real or fake
def define_discriminator(n_inputs):
    optimizer = Adam(0.0002, 0.5)
    model = Sequential()
    model.add(Input(shape=(None, n_inputs),name="input"))
    model.add(Dense(256, name="Dense_Layer"))
    # model.add(Dense(512, input_dim=n_inputs))
    model.add(LeakyReLU(alpha=0.2))


    model.add(Dense(256))  
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.5,name="Dropout_Layer"))


    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return model



# We will define the Gan model after we have define the generator and discriminator models. 
# It is also a sequential model and combine generator with discriminator. 
# NOTE: the discriminator model weight must be not trainable.

def define_gan(generator, discriminator):
    optimizer = Adam(0.0002, 0.5)
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# We will make a plot_history function to visualize the final generator and discriminator loss in the plot.

# create a line plot of loss for the gan and save to file
def plot_history(d_hist, g_hist):
    # plot loss
    plt.subplot(1, 1, 1)
    plt.plot(d_hist, label='d')
    plt.plot(g_hist, label='gen')
    plt.savefig('GanResult.png',bbox_inches = 'tight')
    plt.show()
    plt.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=1000, n_batch=128):
    # determine half the size of one batch, for updating the  discriminator
    half_batch = int(n_batch)
    d_history = []
    g_history = []
    # manually enumerate epochs
      
    b=0
    for epoch in range(n_epochs):
        while(b<2):
            # prepare real samples
            x_real, y_real = generate_real_samples(half_batch)
            # prepare fake examples
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # x_real , x_fake, y_real , y_fake = pre_processing( x_real , x_fake, y_real , y_fake)
    

            # update discriminator
            d_loss_real = d_model.train_on_batch(x_real, y_real)
            d_loss_fake = d_model.train_on_batch(x_fake, y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_history.append(d_loss)
            print(d_loss)
         
            b+=1
        

        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss_fake = gan_model.train_on_batch(x_gan, y_gan)
        print('>%d,  d=%.3f g=%.3f' % (epoch+1, d_loss,  g_loss_fake))
        g_history.append(g_loss_fake)
        b=0
    g_model.save('trained_generated_model_alldata.h5')
    d_model.save('discriminator_alldata.h5')

    plot_history(d_history, g_history)


# We input latent_dim value is 10 to start the training.
data,n_feature=drop_function_Features(data,function_Features)
print(data.head())
print(n_feature)

# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator(n_feature)
# create the generator
generator = define_generator(latent_dim,n_feature)

# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)
