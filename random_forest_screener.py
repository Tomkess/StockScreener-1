import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import util
from time import time
import threading


"""
Alter this list in accordance to what key statistics you want to analyze.
I chose the subset of valuation measures but there is a large array of options
including profitibility, balance sheet, etc.

Reference https://pypi.org/project/yfinance/ to determine the correct formatting
for each key value pair. 

To access use df[6:]
"""
yfinance_statistics = {
    'Market Cap': 'marketCap', 
    'Enterprise Value': 'enterpriseValue', 
    'Trailing P/E': 'trailingPE', 
    'Forward P/E': 'forwardPE', 
    'PEG Ratio': 'pegRatio', 
    'Price/Sales': 'priceToSalesTrailing12Months', 
    'Price/Book': 'priceToBook', 
    'Enterprise Value/Revenue': 'enterpriseToRevenue', 
    'Enterprise Value/EBITDA': 'enterpriseToEbitda'
    }

the_fundamentals = list(yfinance_statistics.keys())

p_changes = [
    'stock_p_change',
    'SP500_p_change'
    ]


def query_to_df(ticker_list, dfs, the_columns):
    """Helper function to propogate dataframe with threading"""
    
    df = pd.DataFrame(columns = the_columns)
    
    for i in range(len(ticker_list)):
        ticker = yf.Ticker(ticker_list[i])
        info = ticker.info
        
        ser = [ticker_list[i], 'N/A', info['regularMarketPrice'], 'N/A']
        for fund in the_fundamentals:
            try:
                ser.append(info[yfinance_statistics[fund]])
            except KeyError:
                ser.append('N/A')
                
        df = df.append(pd.Series(ser, index = the_columns), ignore_index = True)
        print(ticker_list[i], i)
        
    dfs.append(df)
        
    return dfs


def partition(lst, n):
    """Partitions a list into n-sized portions"""
    
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Refactor to allow training feature types and regressor parameters to be set by user
# Bad practice to hard code the list of fundamentals...
def train_classifier():
    """
    Returns random forest regressor trained with historical fundamentals data 
    courtesy of github.com/robertmartin8
    
    The random forest regressor will attempt to predict returns relative to 
    the S&P 500. It is also a common method to use a machine learning "classifier"
    to determine whether a stock will outperform the benchmark by a certain
    standard
    
    For the sake of symmetry to the momentum screener, I want to rank the stocks
    by their predicted returns so I'm doing regression rather than classification.
    """

    url = 'https://raw.githubusercontent.com/robertmartin8/MachineLearningStocks/master/keystats.csv'
    df = pd.read_csv(url, index_col='Date')

    # Removes rows from df that contain 'N/A' or 'nan'
    # Cuts data from ~8700 rows to ~6800
    df = df[the_fundamentals + p_changes].dropna(how = "any").astype(np.float32)

    # Working on solution with better space complexity, but at least it's constant
    # This is a simple solution that keeps the features and labels sets the same size
    training_features = df[the_fundamentals].values
    training_labels = (df['stock_p_change'] - df['SP500_p_change']).tolist()

    # Using default parameters and low estimator count for now
    # Using constant random_state seed so that results are consistent across trials
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(training_features, training_labels)
    
    return regressor


def get_current_data():
    """
    Returns dataframe with current fundamentals data of the first n companies in
    get_sp500_companies(). Yfinance is extremely slow for this type of request,
    currently working on alternative using beautiful soup to parse 
    finance.yahoo.com.
    """
    
    symbols = util.get_sp500_companies()
    
    the_columns = [
        'Symbol', 
        'Number of Shares to Buy',
        'Price',
        'Predicted Relative Returns'
        ] + the_fundamentals

    dfs = []
    threads = []
    for lst in partition(symbols, 10):
        t = threading.Thread(target=query_to_df, args=(lst, dfs, the_columns)) # Creates thread for each list
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    results = pd.concat(dfs)
    
    """
    In case threading breaks...
    
    for i in range(n):
        ticker = yf.Ticker(symbols[i])
        info = ticker.info
        
        df.loc[i, 'Symbol':'Predicted Relative Returns'] = [symbols[i], 'N/A', \
                                            info['regularMarketPrice'], 'N/A']
            
        for fund in the_fundamentals:
            try:
                df.loc[i, fund] = info[yfinance_statistics[fund]]
            except KeyError:
                df.loc[i, fund] = 'N/A'
        
        print(symbols[i])
    """
        
    return results;


def predict_returns(df, regressor, n = 10):
    """Propogates dataframe with predicted returns then returns top n"""
    
    for i in range(len(df)):
        df.loc[i, 'Predicted Relative Returns'] = regressor.predict(\
                            [df.loc[i, 'Market Cap':].values])[0]
            
    df = df.sort_values(by = 'Predicted Relative Returns', ascending = False)[:n]
    
    return df


df = get_current_data()
        
"""
TESTING

df = util.pickle_get()
reg = train_classifier()
df = predict_returns(df, reg, 5)
df = util.num_shares(df, 1000)
"""




    












