import pandas as pd
import pickle

def pickle_obj(obj):
    with open('dict.pickle', 'wb') as f:
        pickle.dump(obj, f)
    
def pickle_get(name):
    pickle_in = open(f'{name}.pickle', 'rb')
    return pickle.load(pickle_in)

def get_sp500_companies():
    """Returns dataframe of all companies and select information"""
    
    all_info = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return all_info[0]['Symbol'].to_list()
    