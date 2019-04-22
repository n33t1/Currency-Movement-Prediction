# Currency-Movement-Prediction
This project used a (Attention-Based) LSTM to predict Foreign Exchange Market(Forex) movement. 

## How To Use
1. Make sure you have Python3. Then run `pip install -r requirements.txt` to install all the libraries. If you are thinking using word vectors, please make sure you have the models (`glove.6B.300d.txt`([GloVe](https://nlp.stanford.edu/projects/glove/)) and/or `GoogleNews-vectors-negative300.bin`([word2vec](https://code.google.com/archive/p/word2vec/))) in `lib/models`.
2. You need both a news title dataset and forex exchange dataset with same timestamps. 

    2.1 News Title Dataset     

    This project used news titles from [All the news](https://www.kaggle.com/snapcrack/all-the-news#articles1.csv) Dataset from Kaggle (ranging from `2016-01-01` to `2017-07-07`). 

    2.2 Forex Dataset 

    This repo used `src/utils/download_forex_data.py` to download **daily foreign exchange rate** for `USD-EUR, USD-CNY, USD-JPY, USD-GBP, USD-BTC` trade-pairs with the Open Exchange Rates API from the same time range as the News Dataset(`2016-01-01` to `2017-07-07`). But you are more than welcome to crawl this info with more accurate timestamps on your own! Just make sure the timestamps of News Title Dataset and Forex Dataset matches up.

    2.3 Knowledge base

    This repo offers a really simple Knowledge Base in `lib/kbase`. Common financial vocabularies are stored in `lib/kbase/common.txt` and trade-pair specific vocabularies are stored in its own text file (for example, `USD-EUR` pair knowledge base are stored in `lib/kbase/EUR.txt`). You are encouraged to expand the KBase. 

3. Run `python src/processing.py` to preprocess the datasets.
4. Use `python src/run.py --trade_pair=['eur','jpy','cny','gbp','btc'] --feature_vec=['word2int','word2vec','word2glove']` to train the model. For example, `python src/run.py --trade_pair=eur --feature_vec=word2int` will train a model for USD-EUR pair with name to entity (word2int) model. 

## Code Structure
```
lib
  |- kbase
     |- *.txt 
  |- models* # YOU NEED TO ADD THESE YOURSELF
     |- glove.6B.300d.txt
     |- GoogleNews-vectors-negative300.bin
  |- raw
     |- forex_16-17.csv
     |- news_titles_16-17.csv
  |- USD-<CURRENCY_NAME>_16-17.csv
src
  |- utils
  |- Attention.py
  |- FeatureVector.py
  |- Model.py
  |- preprocessing.py
  |- run.py
```