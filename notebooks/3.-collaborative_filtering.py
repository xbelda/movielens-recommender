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

# # Bibliography
# - [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)
# - [Getting Started with a Movie Recommendation System](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system#Content-Based-Filtering)
# - [Collaborative Filtering in Pytorch](https://spiyer99.github.io/Recommendation-System-in-Pytorch/)

# cd ..

# %load_ext autoreload
# %autoreload 2

# +
import pandas as pd
from pytorch_lightning.loggers import mlflow
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from sklearn.preprocessing import MinMaxScaler
from src.dataset import MovielensDataset
from src.model import LightningCollaborativeFiltering

# -

# # Data

ratings = pd.read_parquet("./data/processed/ratings.parquet")

ratings.head()

# # Train/Val split
# Since we want to predict future recommendations in our recommender, we will split according to date.

# +
q_timestamp = ratings["Timestamp"].quantile(0.8)

train_idx = ratings["Timestamp"] < q_timestamp
val_idx = ratings["Timestamp"] >= q_timestamp
# -

train_ratings = ratings[train_idx].copy()
val_ratings = ratings[val_idx].copy()

train_ratings.shape

val_ratings.shape

# + [markdown] tags=[]
# # Scaling
# -

scaler = MinMaxScaler()
scaler.fit(train_ratings[["Rating"]])

train_ratings["ScaledRating"] = scaler.transform(train_ratings[["Rating"]]).flatten()
val_ratings["ScaledRating"] = scaler.transform(val_ratings[["Rating"]]).flatten()

# # Dtypes

# Convert Dtypes to Int32
ratings = ratings.astype("Int32")

# # Datasets

BATCH_SIZE = 128
EMBEDDING_DIM = 32
LEARNING_RATE = 1e-3
DROPOUT = 0.2
MAX_EPOCHS = 50

# +
# TODO: Split train/val by date

train_dataset = MovielensDataset(data=train_ratings)
val_dataset = MovielensDataset(data=val_ratings)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
# -

# Callbacks
early_stopping = pl.callbacks.EarlyStopping(monitor="Loss/Val", min_delta=1e-3, patience=3)

model = LightningCollaborativeFiltering(num_users=train_dataset.num_users,
                                        num_movies=train_dataset.num_movies,
                                        embedding_dim=EMBEDDING_DIM,
                                        dropout=DROPOUT,
                                        lr=LEARNING_RATE)

# Auto log all MLflow entities
logger = pl.loggers.MLFlowLogger()

# Train the model
trainer = pl.Trainer(max_epochs=MAX_EPOCHS, gpus=1, callbacks=[early_stopping], logger=logger)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
