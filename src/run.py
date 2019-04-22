import pandas as pd
from random import sample, shuffle, randint
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import datetime
from random import randint
import argparse

from FeatureVector import FeatureVector
from Model import Model

from utils.file_io import open_file, read_file

NEWS_PROCESSED_PATH = "lib/processed_news_titles_16-17.csv"

FOREX_CLASS_TO_INT = {'NL': 1, 'NS': 2, 'PS':3, 'PL':4}
ARG_TO_TRADE_PAIR = {'eur': 'USD-EUR','jpy': 'USD-JPY','cny': 'USD-CNY','gbp': 'USD-GBP','btc': 'USD-BTC'}

def k_fold(dates, k=7):
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

def get_dataset(start, end, fv, labels, is_test):
    feature_vec = fv.fv
    train_x, train_y = [], []
    x = np.array(feature_vec[start:end+1])
    y = np.array(list(map(FOREX_CLASS_TO_INT.get, labels[start: end+1])))
    if len(x.shape) == 4:
        shape = (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = np.reshape(x, shape)

    if is_test:
        baseline_y = np.array([randint(1,4) for _ in range(len(labels[start: end+1]))])
        return x, y, baseline_y
    else:
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
    #labels = df[_trade_pair.upper()].values.tolist()

    date_to_idx = {k: v for v, k in enumerate(dates)}
    serialized_news = df['News'].values.tolist()
    news = [n.split('/,,,/') for n in serialized_news]

    fv = FeatureVector.get_fv(dates, labels, news, feature_vec)

    params = {'MEMORY_SIZE': 100}
    if not is_word_embedding:
        params.update({'VOCAB_SIZE': fv.vocab_size + 1, 'EMBEDDING_SIZE': 128})
    windows = k_fold(dates)
    total_acc, total_baseline_acc = 0, 0
    for s, p, e in windows:
        train_x, train_y = get_dataset(s, p, fv, labels, False)
        test_x, test_y, baseline_y = get_dataset(p, e, fv, labels, True)

        input_size = (None, train_x.shape[1], train_x.shape[2])
        lstm = Model(input_size, is_word_embedding, is_attention, **params)
        history = lstm.train(train_x, train_y, epochs=100)
        
        acc = lstm.evaluate(test_x, test_y)
        baseline_acc = lstm.evaluate(test_x, baseline_y)
        print(f"accuracy: {acc}, baseline_accuracy:{baseline_acc}")
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f"./out/{feature_vec}_{model}_{s}.csv")
        total_acc += acc
        total_baseline_acc += baseline_acc
    
    #avg_acc = total_acc 
    avg_acc = total_acc / len(windows)
    avg_baseline_acc = total_baseline_acc / len(windows)
    with open(f"./out/{feature_vec}_{model}_res.txt", "w+") as f:
        f.write(f"Average accuracy for {feature_vec} is {avg_acc} \n")
        f.write(f"Baseline accuracy for {feature_vec} is {avg_baseline_acc}")
    print(f"Average accuracy for {feature_vec} is {avg_acc}")
    print(f"Baseline accuracy for {feature_vec} is {avg_baseline_acc}")
    

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
                        choices=['lstm','attention_lstm'], 
                        help='Model you want to use.')
    
    args = parser.parse_args()

    trade_pair = args.trade_pair
    feature_vec = args.feature_vec
    model = args.model

    run(trade_pair, feature_vec, model)
