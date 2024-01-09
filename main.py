import pandas as pd 
import random
import numpy as np
import pygame as pg
import tensorflow as tf
import os

WIDTH = 28
HEIGTH = 28
PIXEL_SIZE = 8
FPS = 60

###DECISIONS####

#wanne train again (load=false) or load the last model(load=true)
LOAD = False




#get process id
print('Process-ID: ' + str(os.getpid()))

#Defining the callback function to stop our training once the acceptable accuracy is reached
class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > 0.999):
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True
            
callbacks = myCallback()    

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

train_y = train_dataset['label'].astype('float32')
train_x = train_dataset.drop(['label'], axis = 1).astype('int32')
test_x = test_dataset.astype('float32')
#print(train_x.shape, test_x.shape, train_y.shape)


######reshaping/normalization of x

train_x = train_x.values.reshape(-1,28,28,1)
train_x = train_x/255
test_x = test_x.values.reshape(-1,28,28,1)
test_x = test_x/255        
#print(train_x.shape, test_x.shape, train_y.shape)

######reshaping/categorization of y

train_y = tf.keras.utils.to_categorical(train_y,10)
#print(train_data['label'].head())
#print(self.train_y[0:5,:])

if LOAD == True:
     model=tf.keras.models.load_model("neural_net")

else:
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


    #Compiling and model training with batch size = 50, epochs = 20, and optimizer = adam
    Optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-07,
        name='Adam'
    )

    model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size = 50, epochs = 1, callbacks=[callbacks])

    model.save("neural_net")

            
            

results = model.predict(test_x) 

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

### creates submission.csv for kaggle
#submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
#submission.to_csv("submission.csv",index=False)

comparison = train_dataset
predicted_values = model.predict(train_x)
predicted_values = np.argmax(predicted_values,axis = 1)
predicted_values = pd.Series(predicted_values,name="Label")

comparison = comparison.assign(prediction = predicted_values)
comparison.to_csv("comparison.csv")



pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode((WIDTH*PIXEL_SIZE,HEIGTH*PIXEL_SIZE))
pg.display.set_caption('zufälliges Bild aus MNIST')

def shuffle():   
    rng_row = random.randint(0,len(comparison)-1)
    label = comparison.iloc[rng_row]['label'] 
    prediction = comparison.iloc[rng_row]['prediction']
    while comparison.iloc[rng_row]['label'] == comparison.iloc[rng_row]['prediction']:
        rng_row = random.randint(0,len(comparison)-1)
    
    print('Abweichender Index: ' + str(rng_row))
    pixel_list = []
    color_row = comparison.drop(['label'], axis = 1)
    color_row = color_row.drop(['prediction'], axis = 1)

    for row in color_row.iloc[rng_row]:
        pixel_list.append(color_row.iloc[rng_row][row])
    
    #print(pixel_list)
    return pixel_list, rng_row, prediction, label
    
font = pg.font.SysFont("Consolas", 12)

pixels = shuffle()[0]
rng = shuffle()[1]
label = shuffle()[2]
prediction = shuffle()[3]

run = True
while run:

    clock.tick(FPS)
    screen.fill((0,255,0))

    for i in range(0, WIDTH):
        for j in range(0,HEIGTH):
            #print(i,j, pixels[(i*WIDTH)+j])
            pg.draw.rect(screen, (pixels[(i*WIDTH)+j],pixels[(i*WIDTH)+j],pixels[(i*WIDTH)+j]),(i*PIXEL_SIZE,j*PIXEL_SIZE,PIXEL_SIZE,PIXEL_SIZE))

    img_rng = font.render ("Reihe in train.csv: "+str(rng), True, (255,0,0))
    screen.blit(img_rng, (10,10))
    img_label = font.render ("label: "+str(label), True, (255,0,0))
    screen.blit(img_label, (10,25))
    img_prediction = font.render ("prediction: "+str(prediction), True, (255,0,0))
    screen.blit(img_prediction, (10,40))
    

    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                run = False
            elif event.key == pg.K_SPACE:
                pixels = shuffle()[0]
                rng = shuffle()[1]
                label = shuffle()[2]
                prediction = shuffle()[3]
                         


    pg.display.flip()

pg.quit()