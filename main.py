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

#wanne train again (load=false) or load the last model(load=true) from neural_net file
LOAD_MODE = False
#if true it only show false comparison on scree, else it shows both
ONLY_SHOW_FALSE_MODE = True


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

if LOAD_MODE == True:
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
    model.fit(train_x, train_y, batch_size = 50, epochs = 20, callbacks=[callbacks])

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

comparison['prediction'] = predicted_values

temp_list = []
for i in range(0,len(comparison)):
    if comparison.iloc[i]['label'] == comparison.iloc[i]['prediction']:
        temp_list.append('True')
    else:
        temp_list.append('False')

comparison['compare_state'] = temp_list


pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode((WIDTH*PIXEL_SIZE,HEIGTH*PIXEL_SIZE))
pg.display.set_caption('Compare_Screen')
font = pg.font.SysFont("Consolas", 12)

row = 0
if ONLY_SHOW_FALSE_MODE == True:
    while comparison.iloc[row]['label'] == comparison.iloc[row]['prediction']:
        row += 1

pixels = comparison.iloc[row]
label = comparison.iloc[row]['label']
prediction = comparison.iloc[row]['prediction']
compare_state = comparison.iloc[row]['compare_state']
if compare_state == 'True':
    color = (0,255,0)
else:
    color = (255,0,0)

run = True
while run:

    clock.tick(FPS)
    screen.fill((0,255,0))

    pixel_row = comparison.drop(['label','prediction','compare_state'], axis = 1).iloc[row]
    #print(pixel_row)

    for i in range(0, WIDTH):
        for j in range(0,HEIGTH):
            #print(i,j, pixels[(i*WIDTH)+j])
            #pg.draw.rect(screen, (pixel_row.iloc[(i*WIDTH)+j],pixel_row.iloc[(i*WIDTH)+j],pixel_row.iloc[(i*WIDTH)+j]),(i*PIXEL_SIZE,j*PIXEL_SIZE,PIXEL_SIZE,PIXEL_SIZE))
            pixel_color = (pixel_row.iloc[(i*WIDTH)+j],pixel_row.iloc[(i*WIDTH)+j],pixel_row.iloc[(i*WIDTH)+j])
            pg.draw.rect(screen, pixel_color,(j*PIXEL_SIZE,i*PIXEL_SIZE,PIXEL_SIZE,PIXEL_SIZE))

    img_number = font.render ("Nummer: "+str(row), True, color)
    screen.blit(img_number, (10,10))

    img_label = font.render ("label: "+str(label), True, color)
    screen.blit(img_label, (10,25))

    img_prediction = font.render ("prediction: "+str(prediction), True, color)
    screen.blit(img_prediction, (10,40))

    img_compare_state = font.render ("state: "+str(compare_state), True, color)
    screen.blit(img_compare_state, (10,55))
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                run = False
            elif event.key == pg.K_SPACE:
                if ONLY_SHOW_FALSE_MODE == True:
                    row += 1
                    while comparison.iloc[row]['label'] == comparison.iloc[row]['prediction']:
                        row += 1 
                        if row >= len(comparison):
                            pg.quit()                       
                else:
                    row += 1
                    if row >= len(comparison):
                            pg.quit()   

                pixels = comparison.iloc[row]
                label = comparison.iloc[row]['label']
                prediction = comparison.iloc[row]['prediction']
                compare_state = comparison.iloc[row]['compare_state']

                if compare_state == 'True':
                    color = (0,255,0)
                else:
                    color = (255,0,0)
                         


    pg.display.flip()

pg.quit()