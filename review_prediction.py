"""
predict whether a review is positive or negative
"""

import pandas as pd


# read data to dataframe
fname = 'yelp_training_set/yelp_training_set_review.json'
data = pd.read_json(fname, lines=True)
print(data.shape)
print(data.head())
print()

# check information
print(data['type'].unique())
print()