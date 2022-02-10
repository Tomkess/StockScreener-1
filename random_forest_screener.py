import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import util
import threading
from excel_writer import to_excel
from sklearn.metrics import precision_score


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
                ser.append(np.nan)
                
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
def train_regressor():
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
    
    print("Training regressor...")
    
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
    Old data collection method just in case...
    
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


def predict_returns(df, regressor, n = 50):
    """Propogates dataframe with predicted returns then returns top n"""
    
    df.dropna(inplace = True) # Cuts out tickers with incomplete data
    
    for i in range(10):
        df.loc[i, 'Predicted Relative Returns'] = regressor.predict(\
                                            df.loc[i, 'Market Cap':].values)
            
    df = df.sort_values(by = 'Predicted Relative Returns', ascending = False)[:n]
    
    new_index = np.arange(0, n)
    df.set_index(new_index, inplace = True) # normalizes indices
    
    print("Predicting returns...")
    
    return df


def backtest_random_forest():
    """
    SEVERELY OVERFITTED BACKTEST METHOD
    GIVES EXTREMELY MISLEADING RETURNS
    """

    url = 'https://raw.githubusercontent.com/robertmartin8/MachineLearningStocks/master/keystats.csv'
    df = pd.read_csv(url, index_col='Date')
    
    df.dropna(inplace = True)
    
    X = df[the_fundamentals].values
    Y = (df['stock_p_change'] - df['SP500_p_change']).tolist()
    Z = np.array(df[['stock_p_change', 'SP500_p_change', 'Ticker']])
    
    # Splits dataset
    x_train, x_test, y_train, y_test, z_train, z_test = \
        train_test_split(X, Y, Z, test_size = 0.25, shuffle = True)
    
    # Trains regressor with designated training partition
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(x_train, y_train)
    
    # Creates dictionary of returns, key = original index and value = predicted returns
    predicted_returns = []
    for i in range(len(x_test)):
        predicted_returns.append([i, regressor.predict([x_test[i]])[0]])
    
    # Creates list of ordered tuples from dictionary sorted by predicted returns
    # First value of each tuple corresponds to original position in unsorted list
    sorted_predicted_returns = sorted(predicted_returns, key = lambda x:x[1], reverse = True)
    
    # Iterates through top 50 and compares to actual returns of z_test
    # Averages percentage return
    real_returns = []
    for i in range(50):
        winning_ticker = z_test[sorted_predicted_returns[i][0]]
        real_returns.append(winning_ticker[0] - winning_ticker[1])
        
    real_returns = reject_outliers(real_returns)
    
    total = 0; j = 0
    for i in range(len(real_returns)):
        if real_returns[i] != 0:
            total += real_returns[i]
            j += 1
            
    return total / j
           
    
def reject_outliers(data, m=2):
    """Imperfect method to nullify outliers from an array"""
    
    mean = np.mean(data)
    std = np.std(data)
    
    for i in range(len(data)):
        if data[i] >= (mean + 2*std) or data[i] <= (mean - 2*std):
            data[i] = 0
            
    return data


"""
df = get_current_data()
reg = train_regressor()

util.pickle_obj(df)

df = predict_returns(df, reg)
df = util.num_shares(df, 100000000)

to_excel(df, "Top 50 by Valuation")
"""

x = backtest_random_forest()



    












