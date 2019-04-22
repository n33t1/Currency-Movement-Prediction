from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, TimeDistributed, InputLayer, Flatten
from keras.optimizers import Adam
import keras

import tensorflow as tf
from Attention import Attention

class Model:
    def __init__(self, input_size, is_word_embedding, is_attention, **kargs):
        self.model = self.init_model(input_size, is_word_embedding, is_attention, **kargs)
        
    def init_model(self, input_size, is_word_embedding, is_attention, **kargs):
        print(input_size)
        model = Sequential()
        
        assert('MEMORY_SIZE' in kargs)
        MEMORY_SIZE = kargs['MEMORY_SIZE']

        if not is_word_embedding:
            assert("VOCAB_SIZE" in kargs and 'EMBEDDING_SIZE' in kargs)
            VOCAB_SIZE = kargs['VOCAB_SIZE']
            EMBEDDING_SIZE = kargs['EMBEDDING_SIZE']
            MAX_SEQUENCE_LENGTH = input_size[-1]

            model.add(TimeDistributed(Embedding(input_dim=VOCAB_SIZE, input_length=MAX_SEQUENCE_LENGTH, output_dim=EMBEDDING_SIZE), batch_input_shape=input_size, input_dtype='int32'))
            model.add(TimeDistributed(Flatten()))
        
        model.add(LSTM(MEMORY_SIZE, batch_input_shape=input_size, return_sequences=is_attention, dropout=0.25, recurrent_dropout=0.25))
        if is_attention:
            model.add(Attention(10))
        model.add(Dropout(0.25))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(1, activation = "linear"))
        return model

    def train(self, train_x, train_y, **kwargs):

        epochs = kwargs.get("epochs", 100)
        learning_rate = kwargs.get("learning_rate", 0.0025)
        validation_split = kwargs.get("validation_split", 0.2)
        
        optimizer = Adam(learning_rate)
        self.model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])

        history = self.model.fit(train_x, train_y,
                                    epochs=epochs,
                                    validation_split=validation_split,
                                    verbose=0)
        
        return history

    def evaluate(self, test_x, test_y):
        _, acc = self.model.evaluate(test_x, test_y)
        return acc

    
    def predict(self, test_x):
        res = self.model.predict(test_x)
        return res

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)
