# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# cd ..

import pandas as pd

# # Ratings

ratings = pd.read_csv("./data/raw/ml-1m/ratings.dat", sep='::', header=None, names=["UserID", "MovieID", "Rating", "Timestamp"])

ratings.head()

ratings.to_parquet("./data/processed/ratings.parquet", index=False)

# # Movies

movies = pd.read_csv("./data/raw/ml-1m/movies.dat", sep='::', header=None, names=['MovieID', 'Title', 'Genres'], encoding='iso-8859-1')

movies.head()

movies.to_parquet("./data/processed/movies.parquet", index=False)

# # Users

users = pd.read_csv("./data/raw/ml-1m/users.dat", sep='::', header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='iso-8859-1')

users.head()

users.to_parquet("./data/processed/users.parquet", index=False)
