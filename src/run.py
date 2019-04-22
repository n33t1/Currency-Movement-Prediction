import pandas as pd
from random import sample, shuffle
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import datetime
from random import randint
import argparse

from FeatureVector import FeatureVector
from LSTM import LSTM

from utils.file_io import open_file, read_file

NEWS_PROCESSED_PATH = "lib/processed_news_titles_16-17.csv"
FOREX_PATH = "lib/USD-EUR_forex_16-17.csv"

FOREX_CLASS_TO_INT = {'NL': 1, 'NS': 2, 'PS':3, 'PL':4}
ARG_TO_TRADE_PAIR = {'eur': 'USD-EUR','jpy': 'USD-JPY','cny': 'USD-CNY','gbp': 'USD-GBP','btc': 'USD-BTC'}

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
    return shuffle(res)

def get_dataset(start, end, fv, labels):
    feature_vec = fv.fv
    train_x, train_y = [], []
    x, y = np.array(feature_vec[start:end+1]), np.array(list(map(FOREX_CLASS_TO_INT.get, labels[start: end+1])))
    if len(x.shape) == 4:
        shape = (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = np.reshape(x, shape)
    return x, y

def run(_trade_pair, _feature_vec, model):
    trade_pair = ARG_TO_TRADE_PAIR[_trade_pair]
    feature_vec = f"{trade_pair}_{_feature_vec}"
    is_attention = model == 'attention_lstm'
    is_word_embedding = _feature_vec != 'word2int'

    file_name = open_file(f'lib/{trade_pair}_16-17.csv')
    df = pd.read_csv(file_name)

    dates = df['Date'].values.tolist()
    labels = df['DR_Classes'].values.tolist()

    date_to_idx = {k: v for v, k in enumerate(dates)}
    serialized_news = df['News'].values.tolist()
    news = [n.split('/,,,/') for n in serialized_news]

    fv = FeatureVector.get_fv(dates, labels, news, feature_vec)

    params = {'MEMORY_SIZE': 300}
    if not is_word_embedding:
        params.update({'VOCAB_SIZE': fv.vocab_size + 1, 'EMBEDDING_SIZE': 128})

    windows = k_fold(dates)
    total_acc = 0
    for s, p, e in windows:
        train_x, train_y = get_dataset(s, p, fv, labels)
        test_x, test_y = get_dataset(p, e, fv, labels)
        input_size = (None, train_x.shape[1], train_x.shape[2])
        lstm = BasicLSTM(input_size, is_word_embedding, is_attention, **params)
        lstm.train(train_x, train_y, epochs=10)
        acc = lstm.evaluate(test_x, test_y)
        print("acc: ", acc)
        total_acc += acc
    
    avg_acc = total_acc / len(windows)
    print(f"Average accuracy for {trade_pair}-{feature_vec} is {avg_acc}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trade_pair', 
                        required=True, 
                        choices=['eur','jpy','cny','gbp','btc'], 
                        help='Trade pair you want to use.')

    parser.add_argument('--feature_vec', 
                        required=True, 
                        choices=['word2int','word2vec','word2glove'], 
                        help='Feature vector type you want to use.')
    
    parser.add_argument('--model', 
                        required=True, 
                        choices=['baseline','lstm','attention_lstm'], 
                        help='Model you want to use.')
    
    args = parser.parse_args()

    trade_pair = args.trade_pair
    feature_vec = args.feature_vec
    model = args.model

    run(trade_pair, feature_vec, model)
