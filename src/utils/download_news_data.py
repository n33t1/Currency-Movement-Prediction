import pandas as pd
from file_io import open_files

NEWS_PATH = "lib/raw/all-the-news/*.csv"
TITLE_OUT_PATH = "lib/raw/news_titles_16-17.csv"

def run():
    dataframes = []
    for filename in open_files(NEWS_PATH):
        print(f"Reading {filename}...")

        df = pd.read_csv(filename,
                index_col=[2],
                usecols=[2,3,5,9], 
                header=0,
                parse_dates=['Date'],
                names=["Title", "Publication", "Date", "Content"])
        df = df.loc[df["Publication"].isin(["New York Times", "Reuters", "Washington Post"])]
        df = df['20160101':'20171231']
        dataframes.append(df)

    # format csv
    df = pd.concat(dataframes)
    df = df.drop_duplicates(subset=['Title'], keep='first')
    df = df.sort_values(by=['Date'])

    # saving news titles
    df.pop("Content")
    # df = df.set_index('Date')
    df = df.sort_values(by=['Date'])
    df.to_csv(TITLE_OUT_PATH)
    print(f"News titles saved in {TITLE_OUT_PATH} successfully!")

if __name__ == "__main__":
    run()