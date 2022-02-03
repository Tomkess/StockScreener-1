import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from util import pickle_obj, pickle_get, get_sp500_companies
from time import time
from bs4 import BeautifulSoup
from requests import get


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
    'Trailing PE': 'trailingPE', 
    'Forward PE': 'forwardPE', 
    'PEG Ratio': 'pegRatio', 
    'Price/Sales': 'priceToSalesTrailing12Months', 
    'Price/Book': 'priceToBook', 
    'Enterprise Value/Revenue': 'enterpriseToRevenue', 
    'Enterprise Value/EBITDA': 'enterpriseToEbitda'
    }

the_fundamentals = list(yfinance_statistics.keys())


def train_classifier():
    """
    Returns random forest regressor trained with historical fundamentals data 
    courtesy of github.com/robertmartin8
    
    This is going to apply random forest as a regression technique
    as opposed to classification. The regressor will attempt to predict returns
    rather than identifying "success" or "failure" relative to the S&P 500
    """
    
    fundamentals = the_fundamentals

    p_changes = [
        'stock_p_change',
        'SP500_p_change'
        ]

    url = 'https://raw.githubusercontent.com/robertmartin8/MachineLearningStocks/master/keystats.csv'
    df = pd.read_csv(url)

    # Removes rows from df that contain 'N/A' or 'nan'
    # Cuts data from ~8700 rows to ~6800
    df_clean = df[fundamentals + p_changes].dropna(how = "any").astype(np.float32)

    # Working on solution with better space complexity, but at least it's constant
    # This is a simple solution that keeps the features and labels sets the same size
    training_features = df_clean[fundamentals]
    training_labels = df_clean['stock_p_change'] - df_clean['SP500_p_change']

    # Using default parameters and low estimator count for now
    # Using constant random_state seed so that results are consistent across trials
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(training_features, training_labels)
    
    return regressor

symbols = get_sp500_companies()

the_columns = [
    'Symbol', 
    'Number of Shares to Buy',
    'Price',
    'Predicted Relative Returns'
    ]

all_stats = the_columns + the_fundamentals;

df = pd.DataFrame(columns = all_stats)

t0 = time();

for i in range(len(symbols)):
    ticker = yf.Ticker(symbols[i])
    info = ticker.info
    
    df.loc[i, 'Symbol':'Predicted Relative Returns'] = [symbols[i], 'N/A', \
                                                        info['regularMarketPrice'], 'N/A']
        
    for fund in the_fundamentals:
        df.loc[i, fund] = info[yfinance_statistics[fund]]
    
    print(symbols[i])
    
print(time() - t0);
    












