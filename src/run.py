import pandas as pd
from random import sample
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import datetime
from random import randint
# from MLP import MLP
from FeatureVector import FeatureVector
from utils.file_io import open_files, read_file
from Data import Data

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


NEWS_PROCESSED_PATH = "lib/processed_news_titles_16-17.csv"
FOREX_PATH = "lib/mock_USD-EUR_forex.csv"

# def k_fold(data_size, k = 10):
#     '''Returns a list of test and train data partitioned with k-fold algorithm in the format of [[test_1, train_1], [test_2, train_2]...], where test and train are also list. '''
#     k = data_size if data_size < k else k
#     res = []
#     indexes = sample(range(data_size), k = data_size)
#     partition = [[] for _ in range(k)]
#     for i, idx in enumerate(indexes):
#         partition[i % k].append(idx)

#     for i in range(k):
#         test, train = partition[i], sum(partition[0:i] + partition[i + 1 :k], [])
#         res.append([train, test])
#     return res

def k_fold():
    s, e = datetime.date(2017,1,1), datetime.date(2017,3,5)
    i = randint(0, 180)
    thres = s + datetime.timedelta(i)
    train_start = str(thres - datetime.timedelta(365))
    test_start = str(thres + datetime.timedelta(1))
    train_delta = 365
    test_delta = 60
    return train_start, train_delta, test_start, test_delta
    
# def run_mlp():
#     total_acc = 0
#     feature_vec = mlp_kwargs['feature_vec']
#     k_fold_res = k_fold(feature_vec.data_size)

#     for train_idx, test_idx in k_fold_res:
#         mlp = MultilayerPerceptron(is_dense, input_layer_size, **mlp_kwargs)
#         mlp.train(train_idx, verbose=0)
#         acc = mlp.evaluate(test_idx, is_baseline)
#         total_acc += acc
#     return total_acc / 10

# def classify(train_data, train_labels, test_data, test_labels):
#     print(train_data, train_labels)
#     print(len(train_data), len(train_labels))
#     input_shape = len(train_data)
#     classifier = MLP(input_shape)
#     classifier.train(train_data, train_labels)
#     # mse, pearson_r = classifier.evaluate(test_data, test_labels)
#     print(f"MSE: {mse}\nPearson Correlation: {pearson_r}")
#     # return mse, pearson_r

def get_test_set(s, delta, fv):
    train_x, train_y = [], []
    i = fv.date_to_idx[s]
    x, y = np.array(fv.word_to_vec[i:i + delta]), np.array(fv.labels[i: i + delta])
    shape = (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
    x = np.reshape(x, shape)
    return x, y
    # for i in range(train_delta + 1):
    #     curr_date = str(train_start + datetime.timedelta(i))
    #     dv 
    #     print(curr_date)


def run():
    filename = open_files('tmp/data.obj')[0]
    # # print(filename)
    d = Data.read_obj(filename)
    # # print(d.read_data('2016-01-01'))
    # # print(d.read_data('2016-01-02'))
    # # print(d.read_data('2016-01-03'))
    # # news_df = pd.read_csv(NEWS_PROCESSED_PATH)
    # # forex_df = pd.read_csv(FOREX_PATH)
    fv = FeatureVector.get_feature_vector(d)

    train_start, train_delta, test_start, test_delta = k_fold()
    # print(train_start, train_delta, test_start, test_delta)
    train_x, train_y = get_test_set(train_start, train_delta, fv)
    test_x, test_y = get_test_set(test_start, test_delta, fv)
    # print(train_x, train_y)
    # print(test_x, test_y)

    # train_x, train_y = np.array(fv.word_to_vec), np.array(fv.labels)

    print(train_x, train_x.shape, train_y)

    # def pearson_r(label, prediction):
    #         return tf.contrib.metrics.streaming_pearson_correlation(prediction, label)[1]
    model = Sequential()
    model.add(LSTM(50, batch_input_shape=(None, 5, 90000), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = "linear"))

    model.summary()

    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

    model.fit(train_x, train_y,
            epochs=100,
            validation_split=0.1,
            batch_size=128)

    _, acc = model.evaluate(test_x, test_y)
    print("acc: ", acc)



if __name__ == "__main__":
    run()