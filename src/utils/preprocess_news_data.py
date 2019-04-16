import pandas as pd
import re
import datetime
from file_io import open_files, read_file, save_file
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from string import punctuation
import numpy as np 
from nltk.corpus import sentiwordnet as swn
from heapq import heappush, heappop
# lemmatizer = WordNetLemmatizer()
from Data import Data

NEWS_TITLES_PATH = "lib/news_titles_16-17.csv"
# MOCK_NEWS_TITLES_PATH = "lib/mock_news_titles_16-17.csv"
# OUT_NEWS_PROCESSED_PATH = "lib/processed_news_titles_16-17.csv"
FOREX_PATH = "lib/USD-EUR_forex_16-17.csv"
# FOREX_PATH = "lib/mock_USD-EUR_forex.csv"
KBASE_PATH = "lib/kbase/*.txt"
BAD_WORDS = set(stopwords.words('english') + list(punctuation + '’‘'))
# stemmer = PorterStemmer()

# def penn_to_wn(tag):
#     """
#     Convert between the PennTreebank tags to simple Wordnet tags
#     """
#     if tag.startswith('J'):
#         return wn.ADJ
#     elif tag.startswith('N'):
#         return wn.NOUN
#     elif tag.startswith('R'):
#         return wn.ADV
#     elif tag.startswith('V'):
#         return wn.VERB
#     return None

# def get_sentiment(word,tag):
#     """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """
#     wn_tag = penn_to_wn(tag)
#     if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
#         return []

#     lemma = lemmatizer.lemmatize(word, pos=wn_tag)
#     if not lemma:
#         return []

#     synsets = wn.synsets(word, pos=wn_tag)
#     if not synsets:
#         return []
#     print(synsets)
#     # Take the first sense, the most common
#     synset = synsets[0]
#     swn_synset = swn.senti_synset(synset.name())

#     return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

# def get_tag_synset_notation(tag):
#     """
#     n - NOUN 
#     v - VERB 
#     a - ADJECTIVE 
#     s - ADJECTIVE SATELLITE 
#     r - ADVERB 
#     """
#     if tag.startswith('J'):
#         return 'a'
#     elif tag.startswith('N'):
#         return 'n'
#     elif tag.startswith('R'):
#         return 'r'
#     elif tag.startswith('V'):
#         return 'v'
#     return None
pos_tag_to_synset_notation = {'J': 'a', 'N': 'n', 'V': 'v'}
word_to_synset_cache = {}


# words_data = ['breakdown']
# words_data = [stemmer.stem(x) for x in words_data]
# pos_val = pos_tag(words_data)
# senti_val=[get_sentiment(x,y) for (x,y) in pos_val]
# print(senti_val)

# senti_val=[ get_sentiment(x,y) for (x,y) in pos_val]

def get_ner(text):
    res = []
    ners = ne_chunk(pos_tag(word_tokenize(text)))
    for ner in ners:
        if hasattr(ner, 'label'):
            res.append((ner.label(), ' '.join(c[0] for c in ner)))
    return res

def preprocessing(text):
    ''' Returns normalized text with following modifications:
        - Replace tab or new line characters with space
        - Lowercase words
        - Remove extra spaces
        - Replace punctuation with UNK
        - Replace stop words with UNK
    '''
    text = text.lower()
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    tokens = pos_tag(word_tokenize(text))
    
    valid_tokens = list(filter(lambda token: token[0] not in BAD_WORDS, tokens))
    words, tags = tuple([list(tup) for tup in zip(*valid_tokens)])
    return ' '.join(words), ' '.join(tags)

def get_sentiment(word, notion):
    synsets = list(swn.senti_synsets(word, notion))
    score = sum([s.pos_score() for s in synsets]) - sum([s.neg_score() for s in synsets])
    return score

def get_senti_score(text):
    tokens = pos_tag(word_tokenize(text))
    senti_score = 0
    for word, tag in tokens:
        if word[0].isupper():
            continue 
        elif tag[0] in {'N', 'V', 'J'}:
            if word + tag[0] in word_to_synset_cache:
                senti_score += word_to_synset_cache[word + tag[0]]
            else:
                score = get_sentiment(word, pos_tag_to_synset_notation[tag[0]])
                word_to_synset_cache[word + tag[0]] = score
                senti_score += score
        else:
            continue 
    return senti_score

def select_top_5_news(titles, kbase):
    hq = []
    for t in titles:
        try:
            t = t.split(' - The New York Times')[0]
            set1 = set([w.lower() for w in t.split(' ')])
            hypernyms = set1.intersection(kbase)
            if hypernyms:
                heappush(hq, (1, len(hypernyms), preprocessing(t)))
            else:
                score = get_senti_score(t)
                heappush(hq, (0, abs(score), preprocessing(t)))
        except:
            continue
    res = []
    for _ in range(5):
        _, _, t = heappop(hq)
        res.append(t)
    return res
    

def run():
    # read csv 
    # data = open_files(NEWS_TITLES_PATH)[0]
    d = Data()
    kbase = get_kbase()
    news_filename = open_files(NEWS_TITLES_PATH)[0]
    df = pd.read_csv(news_filename)

    forex_filename = open_files(FOREX_PATH)[0]
    forex_df = pd.read_csv(forex_filename)

    titles = df['Title'].values.tolist()
    s, e = datetime.date(2016,1,1), datetime.date(2017,7,6)
    diff = e - s
    df_dates, df_words = [], []
    for i in range(diff.days + 1):
        curr_date = str(s + datetime.timedelta(i))
        print(curr_date)
        forex_return = forex_df.loc[forex_df['Date'] == curr_date, 'DR_Classes'].tolist()[0]
        if forex_return == 0:
            forex_return = forex_df.loc[forex_df['Date'] == curr_date, 'WR_Classes'].tolist()[0]
        forex_return = 0 if forex_return == -1 else 1
        titles = df.loc[df['Date'] == curr_date, 'Title'].tolist()
        filtered_titles = select_top_5_news(titles, kbase)
        words, _ = zip(*filtered_titles)
        words = list(words)
        d.add_data(curr_date, words, forex_return)
    
    print(d.read_data('2016-01-01'))
    print(d.read_data('2016-01-02'))
    print(d.read_data('2016-01-03'))
    save_file('tmp/data.obj', d)
    # words, tags = [], []

    # for t in titles:
    #     word, tag = preprocessing(t)
    #     words.append(word)
    #     tags.append(tag)

    # df = df.assign(Words=words, Tags=tags)
    
    # df.to_csv(OUT_NEWS_PROCESSED_PATH)
    # print(f"Data saved in {OUT_NEWS_PROCESSED_PATH} successfully!")

    # word_vecs = np.array(get_word2vec(words))
    # save_file('word_vecs', word_vecs)
    # print(f"Data saved in word_vecs successfully!")

def get_kbase():
    names = open_files(KBASE_PATH)
    kbase = set()
    
    for name in names:
        curr = name.split('/')[-1].split('.')[0]
        hypernyms = read_file(name, 'txt')
        kbase.update(hypernyms)
    return kbase

if __name__ == "__main__":
    run()