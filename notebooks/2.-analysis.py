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

movies = pd.read_parquet("./data/processed/movies.parquet")
users = pd.read_parquet("./data/processed/users.parquet")
ratings = pd.read_parquet("./data/processed/ratings.parquet")

# # Movies

movies.head()

len(movies)

print("Ratio of movies that contain year:", movies["Title"].str.contains("\(\d{4}\)").mean())

# # Users

users.head()

users["Gender"].value_counts(normalize=True)

users["Age"].value_counts(normalize=True).sort_index().plot.bar()

users["Zip-code"].nunique()

users["Occupation"].value_counts().sort_index()
