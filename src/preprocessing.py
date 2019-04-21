import pandas as pd
import re
from heapq import heappush, heappop

from string import punctuation
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn


from utils.file_io import open_file, open_files, read_file

FOREX_PATH = "lib/raw/forex_16-17.csv"
NEWS_TITLES_PATH = "lib/raw/news_titles_16-17.csv"
KBASE_PATH = "lib/kbase/*.txt"

CURRENCIES = ["EUR", "JPY", "GBP", "CNY", "BTC"]
BAD_WORDS = set(stopwords.words('english') + list(punctuation + '’‘'))

pos_tag_to_synset_notation = {'J': 'a', 'N': 'n', 'V': 'v'}
word_to_synset_cache = {}

def get_returns(rates):
    ''' Calculate percentage daily returns and weekly returns. '''
    daily_returns, weekly_returns = [0]*7, [0]*7 # place holders
    for i in range(7, len(rates)):
        prev_week, prev_day, curr = rates[i-7], rates[i-1], rates[i]
        daily_return_precentage = 100.0 - curr * 100.0 / prev_day
        weekly_return_precentage = 100.0 - curr * 100.0 / prev_week
        daily_returns.append(daily_return_precentage)
        weekly_returns.append(weekly_return_precentage)
    return daily_returns, weekly_returns

def get_classes(returns):
    ''' Map numercial return values to classes. 
        Range             Class
        =======================
        <= -2.5%           NL
        (0%, -2.5%]        NS
        [0%, 2.5%)         PS
        >= 2.5%            PL
    '''
    classes = []
    for r in returns:
        if r < 0:
            if r < -2.5:
                classes.append('NL')
            else:
                classes.append('NS')
        elif r > 0:
            if r > 2.5:
                classes.append('PL')
            else:
                classes.append('PS')
        else:
            classes.append(pd.NaT) # row will be dropped later
    return classes

def _preprocessing(text):
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
    for i, w in enumerate(words):
        if w.isdigit():
            words[i] = '<NUM>'
    return ' '.join(words), ' '.join(tags)

def _get_sentiment(word, notion):
    synsets = list(swn.senti_synsets(word, notion))
    score = sum([s.pos_score() for s in synsets]) - sum([s.neg_score() for s in synsets])
    return score

def get_senti_score(text):
    ''' Use word net and SentiWordNet to calculate senti score for the given sentence. '''
    tokens = pos_tag(word_tokenize(text))
    senti_score = 0
    for word, tag in tokens:
        if word[0].isupper():
            continue 
        elif tag[0] in {'N', 'V', 'J'}:
            if word + tag[0] in word_to_synset_cache:
                senti_score += word_to_synset_cache[word + tag[0]]
            else:
                score = _get_sentiment(word, pos_tag_to_synset_notation[tag[0]])
                word_to_synset_cache[word + tag[0]] = score
                senti_score += score
        else:
            continue 
    return senti_score

def select_top_n_news(titles, kbase, n=5):
    ''' Using a greedy approach (news_in_kbase, sentiment score) to select top n news. '''
    hq = []
    for t in titles:
        try:
            t = t.split(' - The New York Times')[0]
            set1 = set([w.lower() for w in t.split(' ')])
            hypernyms = set1.intersection(kbase)
            if hypernyms:
                heappush(hq, (1, len(hypernyms), _preprocessing(t)))
            else:
                score = get_senti_score(t)
                heappush(hq, (0, abs(score), _preprocessing(t)))
        except:
            continue
    res = []
    for _ in range(n):
        _, _, t = heappop(hq)
        res.append(t)
    return res

def init_kbase():
    ''' Read hypernyms from knowledge base '''
    names = open_files(KBASE_PATH)
    kbase = {}
    
    for name in names:
        curr = name.split('/')[-1].split('.')[0]
        hypernyms = read_file(name, 'txt')
        kbase[curr] = hypernyms
    return kbase

def get_news(dates, news_df, kbase):
    titles = news_df['Title'].values.tolist()
    res = []
    for curr_date in dates:
        # select top 5 news
        titles = news_df.loc[news_df['Date'] == curr_date, 'Title'].tolist()
        filtered_titles = select_top_n_news(titles, kbase)
        words, _ = zip(*filtered_titles)
        serialized_sents = '/,,,/'.join(list(words))
        res.append(serialized_sents)
    return res

def get_kbase(kbase, pair):
    res = set()
    res.update(kbase['common'])
    res.update(kbase.get(pair, []))
    return res

def run():
    forex_data = open_file(FOREX_PATH)
    forex_df = pd.read_csv(forex_data)

    news_filename = open_file(NEWS_TITLES_PATH)
    news_df = pd.read_csv(news_filename)
    
    kbase = init_kbase()
    
    for curr in CURRENCIES[:1]:
        trade_pair = f"USD-{curr}" 
        print(f"Processing {trade_pair} pair...")
        df = forex_df[['Date', curr]]
        df = df.set_index('Date')
        df = df.sort_values(by=['Date'])

        # calculate numerical values for daily returns and weekly returns
        rates = df[curr].values.tolist()
        daily_returns_percent, weekly_returns_percent = get_returns(rates) # Daily/Weekly returns in percentage
        df = df.assign(Daily_Returns_Percentage=daily_returns_percent, Weekly_Returns_Percentage=weekly_returns_percent)

        # map numerical return values to classes
        dr_classes, wr_classes = get_classes(daily_returns_percent), get_classes(weekly_returns_percent)
 
        df = df.assign(DR_Classes=dr_classes, WR_Classes=wr_classes)
        # drop dates with no FOREX difference
        df = df.dropna()

        print(f"Done preprocessing FOREX data for {trade_pair}, now selecting top 5 news for all the dates...")
        dates = df.index.tolist()
        curr_kbase = get_kbase(kbase, curr)

        serialized_news = get_news(dates, news_df, curr_kbase)
        df = df.assign(News=serialized_news)

        # save dataframes
        filename = f"lib/{trade_pair}_16-17.csv"
        df.to_csv(filename)
        print(f"Data saved in {filename} successfully!")


if __name__ == "__main__":
    run()
