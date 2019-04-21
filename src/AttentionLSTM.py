from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU, Embedding, Input, Permute, Reshape, Multiply
# from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam

from Attention import Attention

class AttentionLSTM:
    def __init__(self, input_size, max_features, maxlen):
        self.model = self.init_model(input_size, max_features, maxlen)

    def attention_3d_block(self, inputs, TIME_STEPS):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
        a = Dense(TIME_STEPS, activation='softmax')(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        # output_attention_mul = concatenate([inputs, a_probs], name='attention_mul', axis=-1)
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul
        
    def init_model(self, input_size, units, maxlen):
        # inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
        inputs = Input(shape=(5, 90000,))
        attention_mul = self.attention_3d_block(inputs, input_size[1])
        lstm_units = 32
        attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
        output = Dense(1, activation='linear')(attention_mul)
        model = Model(input=[inputs], output=output)
        return model

    def train(self, train_x, train_y, **kwargs):
        epochs = kwargs.get("epochs", 500)
        learning_rate = kwargs.get("learning_rate", 0.005)
        validation_split = kwargs.get("validation_split", 0.3)

        optimizer = Adam(learning_rate)
        self.model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])

        self.model.fit(train_x, train_y,
                epochs=4,
                validation_split=validation_split)

