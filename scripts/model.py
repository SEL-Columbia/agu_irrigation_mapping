from keras.models import Sequential, Model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model



class DeepClassifier(Model):
    def __init__(self):
        super(DeepClassifier, self).__init__()
        self.d1 = Dense(8, activation='relu',)
        self.d2 = Dense(8, activation='relu',)
        self.d3 = Dense(8, activation='relu',)
        self.final = Dense(2, activation='softmax')
        self.drop1 = Dropout(0.5)
        self.drop2 = Dropout(0.5)
        self.drop3 = Dropout(0.5)


    def call(self, x):

        x = self.d1(x)
        x = self.drop1(x)
        x = self.d2(x)
        x = self.drop2(x)
        x = self.d3(x)
        x = self.drop3(x)

        return self.final(x)

