import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# # # Generate dummy data
# # x_train = np.random.random((1000, 20))
# # y_train = np.random.randint(2, size=(1000, 1))

# # x_test = np.random.random((100, 20))
# # y_test = np.random.randint(2, size=(100, 1))

# #Generate 2 sets of X variables
# #LSTMs have unique 3-dimensional input requirements 
# seq_length=5
# X =[[i+j for j in range(seq_length)] for i in range(100)]
# X_simple =[[i for i in range(4,104)]]
# X =np.array(X)
# X_simple=np.array(X_simple)

# y =[[ i+(i-1)*.5+(i-2)*.2+(i-3)*.1 for i in range(4,104)]]
# y =np.array(y)
# X_simple=X_simple.reshape((100,1))
# X=X.reshape((100,5,1))
# y=y.reshape((100,1))

# print(X, X.shape)

# # print(y_train, y_test.shape)
# # model = Sequential()
# # model.add(Dense(64, input_dim=20, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(1, activation='sigmoid'))

# model = Sequential()
# # Add an input layer market + news
# #input_size = len(market_prepro.feature_cols) + len(news_prepro.feature_cols)
# # input_shape=(timesteps, input features)
# input_size = 20
# model.add(LSTM(units=128, return_sequences=True, input_shape=(None,input_size)))
# model.add(LSTM(units=64, return_sequences=True ))
# model.add(LSTM(units=32, return_sequences=False))

# # Add an output layer
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# # model.compile(loss='binary_crossentropy',
# #               optimizer='rmsprop',
# #               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           epochs=20,
#           batch_size=128)
# score = model.evaluate(x_test, y_test, batch_size=128)


data = [[[i+j] for i in range(5)] for j in range(100)]
target = [(i+5) for i in range(100)]

data = np.array(data, dtype=float)
target = np.array(target, dtype=float)
# print(data)
# print(data.shape)
# print(target)
# print(target.shape)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)
print(x_train, x_test)
# x_train = np.array([[[0, 1, 2], [1, 2, 3], [3, 4, 5], [4, 5, 6], [5, 6, 7]], [[1, 2, 3], [2, 3, 5], [6, 4, 2], [1, 2, 3], [5, 2, 8]]], dtype=float)
# x_test = np.array([5, 6], dtype=float)
# print(x_train.shape)
# model = Sequential()
# # model.add(Input(batch_shape=(batch_size, timesteps, data_points)))
# model.add(LSTM(1, batch_input_shape=(None, 5, 3), return_sequences=False))
# model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
# model.summary()

# history = model.fit(x_train, x_test, epochs=50)

# results = model.predict(x_test)
# plt.scatter(range(20), results, c="r")
# plt.scatter(range(20), y_test, c="g")
# plt.show()
# plt.plot(history.history["loss"])
# plt.show()
