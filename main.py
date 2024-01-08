import pandas as pd 
import random
import numpy as np
import pygame as pg
from model import Model

WIDTH = 28
HEIGTH = 28
PIXEL_SIZE = 8
FPS = 60



train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

model = Model(train_dataset, test_dataset)
solution = 123456789


# def shuffle(row):
#     random_row = test_dataset.iloc[row]

#     i = 0
#     complete_row = []
#     while i <= 28**2-1:
#         complete_row.append(random_row.iloc[i]) 
#         i += 1

#     grid = np.reshape(complete_row,(28,28))
#     grid = np.transpose(grid)
#     label = solution

#     return grid, label


pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode((WIDTH*PIXEL_SIZE,HEIGTH*PIXEL_SIZE))
pg.display.set_caption('zufälliges Bild aus MNIST')

# row = random.randint(1,len(train_dataset))
# grid = shuffle(row)[0]
# label = shuffle(row)[1]




run = True
while run:

    clock.tick(FPS)
    screen.fill((0,255,0))

    # for i in range(WIDTH):
    #     for j in range(HEIGTH):
    #         pg.draw.rect(screen, (grid[i][j],grid[i][j],grid[i][j]), pg.Rect(i*PIXEL_SIZE, j*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))
    
    # font = pg.font.SysFont(None, 24)
    # img_row = font.render("Reihe: "+str(row), True, (255,0,0))
    # screen.blit(img_row, (10, 10))
    # img_label = font.render("Label: "+str(label), True, (255,0,0))
    # screen.blit(img_label, (10, 25))


    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                run = False
            # if event.key == pg.K_SPACE:
            #     row += 1
            #     grid = shuffle(row)[0]
            #     label = shuffle(row)[1]             


    pg.display.flip()

pg.quit()