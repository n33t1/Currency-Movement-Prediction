import tensorflow as tf
from tensorflow import keras

import numpy as np
np.random.seed(1337) # for reproducibility

import logging # To skip Tensorflow warnings
tf.get_logger().setLevel(logging.ERROR)


class MLP:
    def __init__(self, input_shape, **kwargs):
        self.epoch = kwargs.get('epoch', 200)
        self.model = self.get_model(input_shape) 

    def get_model(self, input_shape):
        # model = keras.Sequential()
        # model.add(keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(9,)))
        # model.add(keras.layers.Dense(64, activation=tf.nn.relu))
        # model.add(keras.layers.Dense(64, activation=tf.nn.relu))
        # model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        # model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        # model.summary()
        # return model

        # vocab_size = 10000

        model = keras.Sequential()
        model.add(keras.layers.Embedding(input_shape, 16))
        model.add(keras.layers.GlobalAveragePooling1D())
        # model.add(keras.layers.Dense(16, input_shape=(1,)))
        model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        model.summary()
        return model
    
    def train(self, batch_X, batch_Y, verbose=0):
        # def pearson_r(label, prediction):
        #     return tf.contrib.metrics.streaming_pearson_correlation(prediction, label)[1] 

        optimizer = tf.keras.optimizers.Adam(0.005)
        print("+++______hi")
        print(batch_X)
        print(batch_Y)
        # self.model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_squared_error", pearson_r])

        self.model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

        # keras.backend.get_session().run(tf.local_variables_initializer()) # for pearson_r to run
        # history = self.model.fit(batch_X,
        #                     batch_Y,
        #                     epochs=self.epoch,
        #                     validation_split=0.1,
        #                     verbose=verbose)
        history = self.model.fit(batch_X,
                    batch_Y,
                    epochs=40,
                    batch_size=3,
                    verbose=1)

    def evaluate(self, test_data, test_labels):
        _, mean_squared_error, pearson_r = self.model.evaluate(test_data, test_labels)
        return mean_squared_error, pearson_r

    def predict(self, batch_X):
        classes = self.model.predict_classes(batch_X)
        return classes
    
    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)
