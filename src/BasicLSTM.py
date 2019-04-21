from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, TimeDistributed, InputLayer, Flatten
from keras.optimizers import Adam

class BasicLSTM:
    def __init__(self, input_size):
        self.model = self.init_model(input_size)
        
    def init_model(self, input_size):
        model = Sequential()
        VOCAB_SIZE = 6624
        EMBEDDING_SIZE = 128
        MEMORY_SIZE = 100
        # model.add(InputLayer(batch_input_shape=input_size))
        model.add(TimeDistributed(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_SIZE), batch_input_shape=input_size, input_dtype='int32'))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(MEMORY_SIZE, return_sequences=True))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dense(1, activation = "linear"))
        # model.add(Embedding(6624, output_dim=128, mask_zero=True))
        # model.add(TimeDistributed(Flatten()))
        # model.add(LSTM(256, batch_input_shape=input_size, return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(128, return_sequences=False))
        # model.add(Dropout(0.2))
        # model.add(Dense(1, activation = "linear"))

        # model.summary()
        return model

    def train(self, train_x, train_y, **kwargs):
        epochs = kwargs.get("epochs", 100)
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
