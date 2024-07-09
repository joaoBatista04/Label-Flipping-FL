import pandas as pd
import tensorflow as tf
import numpy as np

def label_flipping(y_test):
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




