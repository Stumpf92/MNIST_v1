import pandas as pd 
import random
import numpy as np
import pygame as pg

class Model():

    def __init__(self, train_data, test_data):
        print('Model erstellt')

        self.solution = test_data
