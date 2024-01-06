import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def download_data(train_url, dev_url):
    train_df=pd.read_csv(train_url)
    dev_df=pd.read_csv(dev_url)

    train_df[['text1', 'text2']] = train_df['Text'].str.split('\n', 1, expand=True) 
    train_df.drop('Text', axis=1, inplace=True) 

    dev_df[['text1', 'text2']] = dev_df['Text'].str.split('\n', 1, expand=True)
    dev_df.drop('Text', axis=1, inplace=True) 

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    return train_df,val_df, dev_df





