import pandas as pd
import tensorflow as tf
import numpy as np

def label_flipping_old(y_test):
    for label in y_test:
        correct_label = 0
        label = label.numpy()

        for idx, valor in enumerate(label):
            if valor.item() == 1.0:
                correct_label = idx
        
        if correct_label == 0:
            label[0] = 0.0
            label[1] = 1.0

        else:
            label[correct_label] = 0.0
            label[0] = 1.0
        
        label = tf.constant(label)
    
    return y_test

def label_flipping(self, y):
    y_flipped = np.copy(y)  # Cria uma cópia dos rótulos para flipar

    for idx, label in enumerate(y_flipped):
        correct_label = 0
        
        for i in range(len(label)):
            if label[i] == 1.0:
                correct_label = i
        
        if correct_label == 0:
            label[0] = 0.0
            label[1] = 1.0

        else:
            label[correct_label] = 0.0
            label[0] = 1.0
    
    return y_flipped


