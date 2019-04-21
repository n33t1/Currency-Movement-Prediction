import pandas as pd
from random import sample
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import datetime
from random import randint

from BasicLSTM import BasicLSTM
from FeatureVector import FeatureVector

from utils.file_io import open_file, read_file

NEWS_PROCESSED_PATH = "lib/processed_news_titles_16-17.csv"
FOREX_PATH = "lib/USD-EUR_forex_16-17.csv"

FOREX_CLASS_TO_INT = {'NL': 1, 'NS': 2, 'PS':3, 'PL':4}


def k_fold(dates, k=5):
    ''' Create k windows. For example, if k=5 and there are 100 dates, we create 5 windows, where window 1 has train=dates[0:50] and test=[50:60], window 2 has train=dates[10:60] and test=[60:70], so on so forth. '''
    n = len(dates)
    delta = n // (k * 2)
    pivot = delta * k
    s, p, e = 0, pivot, pivot + delta
    res = []
    for i in range(k):
        res.append((s, p, e))

        s += delta
        p += delta 
        e += delta
    return res

def get_dates_range(s, delta):
    res = []
    for i in range(delta):
        curr_date = s + datetime.timedelta(i)
        res.append(curr_date)
    return res

def test(test_x, test_y):
    _, acc = model.evaluate(test_x, test_y)
    return acc

def get_dataset(start, end, feature_vec, labels):
    train_x, train_y = [], []
    x, y = np.array(feature_vec[start:end+1]), np.array(list(map(FOREX_CLASS_TO_INT.get, labels[start: end+1])))
    if len(x.shape) == 4:
        shape = (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = np.reshape(x, shape)
    return x, y

if __name__ == "__main__":
    file_name = open_file('lib/USD-EUR_16-17.csv')
    df = pd.read_csv(file_name)

    dates = df['Date'].values.tolist()
    labels = df['DR_Classes'].values.tolist()

    date_to_idx = {k: v for v, k in enumerate(dates)}
    serialized_news = df['News'].values.tolist()
    news = [n.split('/,,,/') for n in serialized_news]

    fv = FeatureVector.get_fv(dates, labels, news, 'USD-EUR_word2int')
    feature_vec = fv.fv

    windows = k_fold(dates)
    for s, p, e in windows[:1]:
        train_x, train_y = get_dataset(s, p, feature_vec, labels)
        test_x, test_y = get_dataset(p, e, feature_vec, labels)
        input_size = (None, train_x.shape[1], train_x.shape[2])
        lstm = BasicLSTM(input_size)

        lstm.train(train_x, train_y, epochs=500)
        lstm.save_weights('tmp/basic_lstm_1000.weights')
        acc = lstm.evaluate(test_x, test_y)
        print("acc: ", acc)
