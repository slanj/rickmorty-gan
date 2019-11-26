import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop


class Model:
    def __init__(self):
        self.model = None
        self.maxlen = 200
        self.chars = []

    def create_model_v1(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.maxlen, len(self.chars))))
        self.model.add(Dense(len(self.chars), activation='softmax'))

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))


if __name__ == '__main__':
    m = Model()
    m.chars = [1, 2, 3]
    m.create_model_v1()
    m.compile()