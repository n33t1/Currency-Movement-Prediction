from utils.file_io import open_files, read_file, save_file
import numpy as np 
from Data import Data

class FeatureVector:
    def __init__(self, data):
        self.labels = []
        self.dates = []
        self.date_len = 0
        self.date_to_idx = {}
        self.word_to_ints = []
        self.word_to_vec = []
        self.max_seq_len = 0
        self.vocab = set()
        self.idx_to_word = []
        self.word_to_int = {}
        self.group_timestamp(data)
    
    # @classmethod
    # def get_feature_vector(cls, news_df, forex_df):
    #     path = 'tmp/feature_vector.obj'
    #     try:
    #         filename = open_files(path)
    #         if not filename:
    #             raise FileNotFoundError
    #         feature_vec = read_file(filename[0], 'obj')
    #     except FileNotFoundError:
    #         feature_vec = FeatureVector(news_df, forex_df)
    #         save_file(path, feature_vec)
    #     return feature_vec
    @classmethod
    def get_feature_vector(cls, data):
        path = 'tmp/feature_vector.obj'
        try:
            filename = open_files(path)
            if not filename:
                raise FileNotFoundError
            feature_vec = read_file(filename[0], 'obj')
        except FileNotFoundError:
            feature_vec = FeatureVector(data)
            save_file(path, feature_vec)
        return feature_vec

    # def group_timestamp(self, news_df, forex_df):
    #     dates = forex_df['Date'].values.tolist()
    #     model = self._get_word2vec_model('lib/models/GoogleNews-vectors-negative300.bin')

    #     for date in dates:
    #         label = forex_df.loc[forex_df['Date'] == date, 'DR_Classes'].tolist()[0]
    #         self.labels.append(label)
    #         news = news_df.loc[news_df['Date'] == date, 'Words'].tolist()
    #         self.max_seq_len = max(self.max_seq_len, len(news))
    #         self.word_to_vec.append([])

    #         # for text in news[:10]:
    #         #     temp = []
    #         #     for word in text.split()[:3]:
    #         #         if word in self.vocab:
    #         #             temp.append(self.word_to_int[word])
    #         #         else:
    #         #             self.idx_to_word.append(word)
    #         #             self.vocab.add(word)
    #         #             self.word_to_int[word] = len(self.idx_to_word) - 1
    #         #             temp.append(self.word_to_int[word])
    #         #     self.word_to_ints[-1].append(temp)
    #         for text in news[:10]:
    #             words = text.split()
    #             _word_vecs = self.get_word2vec(words, model)
    #             self.word_to_vec[-1].append(_word_vecs)

    def group_timestamp(self, d):
        dates = d.date
        model = self._get_word2vec_model('lib/models/GoogleNews-vectors-negative300.bin')

        for i, date in enumerate(dates):
            label = d.label[i]
            news = d.news[i]
            self.dates.append(date)
            self.labels.append(label)
            self.date_len += 1
            self.date_to_idx[date] = self.date_len - 1
            # news = news_df.loc[news_df['Date'] == date, 'Words'].tolist()
            # self.max_seq_len = max(self.max_seq_len, len(news))
            self.word_to_vec.append([])

            # for text in news:
            #     temp = []
            #     for word in text.split()[:3]:
            #         if word in self.vocab:
            #             temp.append(self.word_to_int[word])
            #         else:
            #             self.idx_to_word.append(word)
            #             self.vocab.add(word)
            #             self.word_to_int[word] = len(self.idx_to_word) - 1
            #             temp.append(self.word_to_int[word])
            #     self.word_to_ints[-1].append(temp)
            for text in news:
                words = text.split(' ')
                _word_vecs = self.get_word2vec(words, model)
                self.word_to_vec[-1].append(_word_vecs)
    
    def _get_word2vec_model(self, path):
        print("Loading word2vec...")
        import gensim 
        from gensim.models import Word2Vec 
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        print("Done loading! Now creating word vectors...")
        return model

    def get_word2vec(self, words, model):
        n = len(words)
        temp = np.zeros((300, 300), dtype=np.float32)
        for word in words:
            if word not in model.vocab:
                continue
            else:
                vec = model.get_vector(word)
                temp = np.add(temp, vec)
        return np.true_divide(temp, n)
