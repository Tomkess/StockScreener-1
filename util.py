import pandas as pd
import pickle
import math


def pickle_obj(obj):
    with open('dict.pickle', 'wb') as f:
        pickle.dump(obj, f)
    
    
def pickle_get():
    pickle_in = open('dict.pickle', 'rb')
    return pickle.load(pickle_in)


def get_sp500_companies():
    """Returns dataframe of all companies and select information"""
    
    all_info = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return all_info[0]['Symbol'].to_list()


# What is the best portfolio weighting strategy?
def num_shares(winners, portfolio_value):
    """Calculates number of shares/ticker for an even-weighted portfolio"""
    
    dollars_per_ticker = portfolio_value / len(winners)
    for row in winners.index:
        winners.loc[row, 'Number of Shares to Buy'] = \
            math.floor(dollars_per_ticker / winners.loc[row, 'Price'])
            
    return winners