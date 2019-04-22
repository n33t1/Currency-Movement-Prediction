import numpy as np 
from itertools import zip_longest

from utils.file_io import open_file, read_file, save_file, exist_file

GLOVE_PATH = 'lib/models/glove.6B.300d.txt.word2vec'
GLOVE_RAW = 'lib/models/glove.6B.100d'
#WORD2VEC_PATH = "/Users/kuriko/Documents/lib/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
WORD2VEC_PATH = "/home/kuriko/lib/GoogleNews-vectors-negative300.bin.gz"
#WORD2VEC_PATH = 'lib/models/GoogleNews-vectors-negative300.bin'

class FeatureVector:
    def __init__(self, dates, labels, news, type_):
        self.x = len(dates)
        self.y = len(news[0])
        self.z = max([max([len(sent.split(' ')) for sent in n]) for n in news])
        if type_ == 'word2int':
            self.fv = self.init_word2int(dates, labels, news)
        elif type_ == 'word2vec':
            self.fv = self.init_word2vec(dates, labels, news, "Word2Vec")
        elif type_ == 'word2glove':
            self.fv = self.init_word2vec(dates, labels, news, "GloVE")

    @classmethod
    def get_fv(cls, dates, labels, news, file_name):
        path = f'tmp/{file_name}.obj'
        try:
            filename = open_file(path)
            feature_vec = read_file(filename, 'obj')
        except Exception:
            type_ = file_name.split('_')[1]
            feature_vec = FeatureVector(dates, labels, news, type_)
            save_file(path, feature_vec)
        return feature_vec


    def init_word2int(self, dates, labels, news):
        word_index = {'<PAD>': 0, '<NUM>': 1}
        word_set = set(['<PAD>', '<NUM>'])
        
        N = 2
        res = []

        for i, date in enumerate(dates):
            sents = news[i]
            day_vec = []
            for text in sents:
                curr_vec = np.zeros(self.z, dtype=np.int32)
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
        self.vocab_size = len(word_set)
        return res
    
    def _get_word2vec_model(self, path, type_):
        if type_ == 'GloVE':
            print("Loading Stanford’s GloVe Embedding model...")
            if not exist_file(path):
                if not exist_file(GLOVE_RAW):
                    raise Exception("Please download GloVE model and make sure it exist in {GLOVE_RAW}!")
                print("Initializing Stanford’s GloVe Embedding model...")
                from gensim.scripts.glove2word2vec import glove2word2vec
                word2vec_output_file = 'lib/models/glove.6B.100d.txt.word2vec'
                glove2word2vec(path, word2vec_output_file)

            from gensim.models import KeyedVectors
            model = KeyedVectors.load_word2vec_format(path, binary=False)
            print("Done loading! Now creating word vectors...")
            return model
        else:
            if not exist_file(path):
                raise Exception("Please download Word2Vec model and make sure it exist in {path}!")
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

    def init_word2vec(self, dates, labels, news, vec_type):
        model_path = GLOVE_PATH if vec_type == 'GloVE' else WORD2VEC_PATH
        model = self._get_word2vec_model(model_path, vec_type)
        
        res = []
        for i, date in enumerate(dates):
            sents = news[i]
            res.append([])
            for text in sents:
                words = text.split(' ')
                _word_vecs = self.get_word2vec(words, model)
                res[-1].append(_word_vecs)
        return np.array(res)
