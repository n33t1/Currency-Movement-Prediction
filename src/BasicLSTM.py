from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.optimizers import Adam

class BasicLSTM:
    def __init__(self, input_size):
        self.model = self.init_model(input_size)
        
    def init_model(self, input_size):
        model = Sequential()
        model.add(LSTM(256, batch_input_shape=input_size, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation = "linear"))

        # model.summary()
        return model

    def train(self, train_x, train_y, **kwargs):
        epochs = kwargs.get("epochs", 500)
        learning_rate = kwargs.get("learning_rate", 0.005)
        validation_split = kwargs.get("validation_split", 0.3)

        optimizer = Adam(learning_rate)
        self.model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])

        history = self.model.fit(train_x, train_y,
                                    epochs=epochs,
                                    validation_split=validation_split)

    def evaluate(self, test_x, test_y):
        _, acc = self.model.evaluate(test_x, test_y)
        return acc
    
    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)
