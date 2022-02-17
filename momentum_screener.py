import pandas as pd
import yfinance as yf
from scipy import stats
import math
import util


# Yahoo Finance API breaks for period = '1y'...
def download_pricing_data(symbols, period = '1y'):
    """Downloads all pricing data for given symbols"""
    
    symbols_str = ' '.join(symbols)
    data = yf.download(tickers = symbols_str, period = period, \
                       group_by = 'ticker', threads = True)  
    return data


def returns(curr_price, days_ago, prices):
    """Returns change in price since days_ago as a percentage"""
    
    return (curr_price - prices[days_ago])/prices[days_ago]


# Alter this dictionary for any unique periods
def get_index(period, length):
    """Returns index for ticker_prices"""
    
    # More elegant way to calculate days?
    dict = {
        '1Y': 0,
        '6M': int(length/2),
        '3M': 3 * int(length/4),
        '1M': 11 * int(length/12)
        }
    
    return dict[period]


def get_all_returns(symbol, periods):
    """Returns price returns for a specific ticker across all fields"""
    
    try:
        # Using prices at 'Open' rather than 'Adj Close' which is also common
        returns = []
        ticker_prices = yf.Ticker(symbol).history(period = periods[0])['Open']
        price = ticker_prices[-1]; returns.append(price)
        
        for period in periods:
            old_price = ticker_prices[get_index(period, len(ticker_prices))]
            returns.append((price - old_price)/old_price)
    
    # Pricing info is missing for some Class B stocks
    except IndexError:
        return [0] * (len(periods)+1)
     
    return returns


def get_momentum_data(periods = ['1Y', '6M', '3M']):
    """Propogates dataframe w/ momentum data of all stocks"""
    
    print("Obtaining momentum data...")
    
    symbols = util.get_sp500_companies()

    # Creating output dataframe
    the_columns = [
        'Symbol',
        'Number of Shares to Buy',
        'Price',
        'Percentile Avg.'
        ]
    
    # Dataframe size and fields change depending on input
    for period in periods:
        the_columns.append(f'{period} Returns')
        the_columns.append(f'{period} Returns Percentile')
        
    df = pd.DataFrame(columns = the_columns)
    
    i = 0 # testing purposes
    for symbol in symbols:
        # Break statement for testing purposes
        """
        if i == 3:
            break
        """
        
        # Obtaining returns and appending to dataframe
        returns = get_all_returns(symbol, periods)
        fields = [symbol, 'N/A', returns[0], 'N/A']
        
        for j in range(1, len(returns)):
            fields.append(returns[j])
            fields.append('N/A')
        
        df = df.append(
            pd.Series(fields, index = the_columns), ignore_index = True)
        
        # Print statement to ensure things are actually working
        # Also, it's incredibly boring to wait for the dataframe to fill
        # while nothing happens on your screen
        print(symbol, i); 
        i += 1
        
    return df


# Refactor
def get_winners(df, n = 50, periods = ['1Y', '6M', '3M']):    
    """Returns dataframe with the top n tickers by percentile avg."""

    # Generates percentiles for each time range and average
    for row in df.index:
        total = 0
        for period in periods:
            spec = f'{period} Returns'
            
            # Iterating through percentile columns
            percentile = stats.percentileofscore(df[spec].to_list(), df.loc[row, spec])
            total += percentile
            
            df.loc[row, f'{spec} Percentile'] = percentile
        
        # Appends average to 'Percentile Avg.' column
        df.loc[row, 'Percentile Avg.'] = total / 3
        
    # Slices and returns top n rows in dataframe
    df = df.sort_values(by = 'Percentile Avg.', ascending = False)[:n]
    return df


# What is the best portfolio weighting strategy?
def num_shares(winners, portfolio_value):
    """Calculates number of shares/ticker for an even-weighted portfolio"""
    
    dollars_per_ticker = portfolio_value / len(winners)
    for row in winners.index:
        winners.loc[row, 'Number of Shares to Buy'] = \
            math.floor(dollars_per_ticker / winners.loc[row, 'Price'])
            
    return winners


if __name__ == '__main__':
    df = get_momentum_data()
    
    top_n = int(input("# of stocks in portfolio: "))
    winners = get_winners(df, n = top_n)
    
    portfolio_value = float(input("Portfolio value: "))
    winners = num_shares(winners, portfolio_value)
    
    winners.to_excel('Top 50 by Momentum')
    print("Done!")
    
    # pickle_obj(winners)


