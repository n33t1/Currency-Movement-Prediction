# Currency-Movement-Prediction
This project is a LSTM-based Recurrent Neural Network that predicts currency exchange market variations based on financial/political news articles. The articles will be preprocessed (format cleanup, entity extraction, etc) and then be fed into the neural network. The expected output of this neural network is the predicted variations of the major currencies based on the news article(s) given. 

## How To Use
1. Make sure you have Python3. Then run `pip install -r requirements.txt` to install all the libraries. If you are thinking using word vectors, please make sure you have the models in `lib/models`. 
2. Run `python src/processing.py` to preprocess the datasets.
3. Use `python src/run.py --trade_pair=['eur','jpy','cny','gbp','btc'] --feature_vec=['word2int','word2vec','word2glove']` to train the model. For example, `python src/run.py --trade_pair=eur --feature_vec=word2int` will train a model for USD-EUR pair with name to entity (word2int) model. 
