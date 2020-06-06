import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import os
import glob

SAMPLE_PATH = 'data/order_books/2012/IF1201.csv'
#'data/order_books/2011/IF1101.csv'

def read(path, is_buy=False, is_dp=True, no_obs=None, slice_size=50):
    df = pd.read_csv(path).iloc[:, 1:].head(no_obs)
    price = (df['S1'] + df['B1']) / 2
    df['price'] = price
    num_slices = int(no_obs / slice_size) if no_obs else int(df.shape[0] / slice_size)
    if is_dp:
        new_df = pd.DataFrame(df, columns=['S5', 'S4', 'S3', 'S2', 'S1', 'price', 'B5', 'B4', 'B3', 'B2', 'B1', 'SV5', 'SV4', 'SV3', 'SV2', 'SV1', 'BV1', 'BV2', 'BV3', 'BV4', 'BV5', 'price'])
    elif is_buy:
        new_df = pd.DataFrame(df, columns=['B1', 'B2', 'B3', 'B4', 'B5', 'BV1', 'BV2', 'BV3', 'BV4', 'BV5', 'price'])
    else:   
        new_df = pd.DataFrame(df, columns=['S1', 'S2', 'S3', 'S4', 'S5', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'price'])
    return [new_df.iloc[n*slice_size:(n+1)*slice_size, :] for n in range(num_slices)]
        
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
