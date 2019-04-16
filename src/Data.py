import pickle
import os

class Data: 
    def __init__(self):
        self.idx = set()
        self.date_to_idx = {}
        self.date_len = 0

        self.date = []
        self.news = []
        self.label = []
    
    def add_data(self, date, news, label):
        self.idx.add(date)

        self.date.append(date)
        self.news.append(news)
        self.label.append(label)

        self.date_len += 1

        self.date_to_idx[date] = self.date_len - 1

    def read_data(self, date):
        assert date in self.idx
        idx = self.date_to_idx[date]
        return self.date[idx], self.news[idx], self.label[idx]

    @classmethod
    def read_obj(cls, name):
        with open(name, 'rb') as f:
            obj = pickle.load(f)
            return obj
