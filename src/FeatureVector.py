from utils.file_io import open_files, read_file, save_file
import numpy as np 
from itertools import zip_longest

from DataVector import DataVector

class FeatureVector(DataVector):
    def __init__(self, data_vector):
        super(FeatureVector, self).__init__(**data_vector.__dict__)
        self.data_vec = data_vector
        self.word2vec = self.init_word2vec()
        print(self.word2vec)
        
        # self.word2int = self.init_word2int()
    
    # def init_feature_vector(self):
    #     self.get_name2entity()
    
    def init_word2int(self):
        word_index = {'<PAD>': 0, '<NUM>': 1}
        word_set = set(['<PAD>', '<NUM>'])

        N = 2
        res = []

        for i, date in enumerate(self.dates):
            news = self.news[i]
            day_vec = []
            for text in news:
                curr_vec = np.zeros(self.longest_wv, dtype=np.int32)
                words = text.split(' ')
                for i, w in enumerate(words):
                    if w not in word_set:
                        word_set.add(w)
                        N += 1
                        idx = N - 1
                        word_index[w] = idx
                    else:
                        idx = word_index[w]
                    curr_vec[i] = idx
                day_vec.append(curr_vec)
            res.append(np.array(day_vec))
        res = np.array(res)
        return res
    
    # def _get_word2vec_model(self, path):
    #     print("Loading word2vec...")
    #     import gensim 
    #     from gensim.models import Word2Vec 
    #     model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    #     print("Done loading! Now creating word vectors...")
    #     return model

    def _get_word2vec_model(self, path):
        print("Loading Stanford’s GloVe Embedding model...")
        # from gensim.scripts.glove2word2vec import glove2word2vec
        # word2vec_output_file = 'lib/models/glove.6B.100d.txt.word2vec'
        # glove2word2vec(path, word2vec_output_file)

        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format(path, binary=False)
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

    def init_word2vec(self):
        # model = self._get_word2vec_model('lib/models/GoogleNews-vectors-negative300.bin')
        model = self._get_word2vec_model('lib/models/glove.6B.300d.txt.word2vec')
        
        res = []
        for i, date in enumerate(self.dates):
            news = self.news[i]
            res.append([])
            for text in news:
                words = text.split(' ')
                _word_vecs = self.get_word2vec(words, model)
                res[-1].append(_word_vecs)
        return np.array(res)

