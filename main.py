import pandas as pd 
import random
import pygame as pg

WIDTH = 28
HEIGTH = 28
PIXEL_SIZE = 8
FPS = 60



train_dataset = pd.read_csv('train.csv')
#print(train_dataset)

random_row = train_dataset.iloc[random.randint(1,len(train_dataset))]
print(random_row)
print(random_row[2])


pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode((WIDTH*PIXEL_SIZE,HEIGTH*PIXEL_SIZE))
pg.display.set_caption('zufälliges Bild aus MNIST')



run = True
while run:

    clock.tick(FPS)
    screen.fill((0,0,0))

    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                run = False


    pg.display.flip()

pg.quit()