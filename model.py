import pandas as pd 
import random
import numpy as np
import pygame as pg
import tensorflow as tf

class Model():

    def __init__(self, train_data, test_data):
        
        self.train_y = train_data['label'].astype('float32')
        self.train_x = train_data.drop(['label'], axis = 1).astype('int32')
        self.test_x = test_data.astype('float32')
        print(self.train_x.shape, self.test_x.shape, self.train_y.shape)


        ######reshaping/normalization of x

        self.train_x = self.train_x.values.reshape(-1,28,28,1)
        self.train_x = self.train_x/255
        self.test_x = self.test_x.values.reshape(-1,28,28,1)
        self.test_x = self.test_x/255        
        print(self.train_x.shape, self.test_x.shape, self.train_y.shape)

        ######reshaping/categorization of y

        self.train_y = tf.keras.utils.to_categorical(self.train_y,10)
        #print(train_data['label'].head())
        #print(self.train_y[0:5,:])

        ######Model Parameters
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32,(3,3),activation = 'relu', input_shape=(28,28,1)),
            tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
            tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
            tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            #tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(10, activation='softmax')
            ])
        print(model.summary())

        



        #Defining the callback function to stop our training once the acceptable accuracy is reached
        class myCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    if(logs.get('accuracy') > 0.999):
                        print("\nReached 99.9% accuracy so cancelling training!")
                        self.model.stop_training = True
            
        callbacks = myCallback()    

        #Compiling and model training with batch size = 50, epochs = 20, and optimizer = adam
        Optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-07,
            name='Adam'
        )

        model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.train_x, self.train_y, batch_size = 50, epochs = 20, callbacks=[callbacks])

        
        