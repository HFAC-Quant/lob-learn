import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import os
import glob

SAMPLE_PATH = 'data/order_books/2011/IF1101.csv'

def read_scale(path, no_obs=None, slice_size=50):
    df = pd.read_csv(path).iloc[:, 1:].head(no_obs)
    a = np.hstack([df.shift(x) for x in range(slice_size)])[slice_size:,]
    return QuantileTransformer().fit_transform(a)

def read_bid_ask(path, no_obs=None):
    df = pd.read_csv(path).iloc[:, 1:].head(no_obs)
    return df['B1'], df['S1']

def generate_data(function, *args, **kwargs):
    '''Yields data one at a time'''
    
    # Find all csv files in data/order_books
    os.chdir( 'data/order_books/' )
    PATHS = ['data/order_books/' +  x
              for x in glob.glob( '*/**.csv' )]
    os.chdir( '../..' )
    for t in PATHS:
        yield function(t, *args, **kwargs)
