import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import os
import glob


def read(path, is_buy=True, is_dp=True, slice_size=50, num_slices=50):
    #path.seek(0, os.SEEK_END)
    #print(path)
    #totallen = open(path, 'r').tell()
    totallen = len(pd.read_csv(path).iloc[:, 1:])
    df = pd.read_csv(path, skiprows=range(1, totallen - slice_size*num_slices)).iloc[:, 1:]
    #print(len(df))
    #print(df.head())
    #df = pd.read_csv(path).iloc[:, 1:].head(no_obs)
    price = (df['S1'] + df['B1']) / 2
    df['price'] = price
    # num_slices = int(no_obs / slice_size) if no_obs else int(df.shape[0] / slice_size)
    new_df = pd.DataFrame(df, columns=['B5', 'B4', 'B3', 'B2', 'B1', 'price', 'S1', 'S2', 'S3', 'S4', 'S5', 'BV5',
                                           'BV4', 'BV3', 'BV2', 'BV1', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'price'])
        #new_df = pd.DataFrame(df, columns=['B1', 'B2', 'B3', 'B4', 'B5', 'BV1', 'BV2', 'BV3', 'BV4', 'BV5', 'price'])
        #new_df = pd.DataFrame(df, columns=['S1', 'S2', 'S3', 'S4', 'S5', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'price'])
    return [new_df.iloc[n*slice_size:(n+1)*slice_size, :] for n in range(num_slices)]
        
def read_scale(path, slice_size=50):
    df = pd.read_csv(path).iloc[:, 1:]
    a = np.hstack([df.shift(x) for x in range(slice_size)])[slice_size:,]
    return QuantileTransformer().fit_transform(a)

def read_bid_ask(path):
    df = pd.read_csv(path).iloc[:, 1:]
    return df['B1'], df['S1']

def generate_data(function, path=None, *args, **kwargs):
    '''Yields data one at a time'''

    if path is None:
        # find all csv files in data/order_books
        os.chdir( 'data/order_books/' )
        PATHS = ['data/order_books/' +  x
                   for x in glob.glob( '*/**.csv' )]
    else:
        PATHS = [path]
    print(f"PATHS: {sorted(PATHS)}")
    # os.chdir( '../..' )
    for t in sorted(PATHS):
        yield function(t, *args, **kwargs)