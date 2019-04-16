import pandas as pd
from file_io import open_files

FOREX_PATH = "lib/raw/forex_16-17.csv"
CURRENCIES = ["EUR", "JPY", "GBP", "CNY", "BTC"]

def get_returns(rates):
    daily_returns, weekly_returns = [0]*7, [0]*7 # place holders
    for i in range(7, len(rates)):
        prev_week, prev_day, curr = rates[i-7], rates[i-1], rates[i]
        daily_return_precentage = 100.0 - curr * 100.0 / prev_day
        weekly_return_precentage = 100.0 - curr * 100.0 / prev_week
        daily_returns.append(daily_return_precentage)
        weekly_returns.append(weekly_return_precentage)
    return daily_returns, weekly_returns

def get_classes(returns):
    classes = []
    for r in returns:
        if r < 0:
            classes.append(-1)
        elif r > 0:
            classes.append(1)
        else:
            classes.append(0)
    return classes

def run():
    forex_data = open_files(FOREX_PATH)[0]
    forex_df = pd.read_csv(forex_data)

    for curr in CURRENCIES:
        name = f"USD-{curr}" 
        print(f"Processing {name} pair...")
        df = forex_df[['Date', curr]]
        df = df.set_index('Date')
        df = df.sort_values(by=['Date'])

        rates = df[curr].values.tolist()
        daily_returns_percent, weekly_returns_percent = get_returns(rates) # Daily/Weekly returns in percentage
        df = df.assign(Daily_Returns_Percentage=daily_returns_percent, Weekly_Returns_Percentage=weekly_returns_percent)

        dr_classes, wr_classes = get_classes(daily_returns_percent), get_classes(weekly_returns_percent)
        df = df.assign(DR_Classes=dr_classes, WR_Classes=wr_classes)
        
        df = df[7:]

        filename = f"lib/{name}_forex_16-17.csv"
        df.to_csv(filename)
        print(f"Data saved in {filename} successfully!")

if __name__ == "__main__":
    run()
