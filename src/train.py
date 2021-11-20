# Bibliography
# - [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)
# - [Getting Started with a Movie Recommendation System](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system#Content-Based-Filtering)
# - [Collaborative Filtering in Pytorch](https://spiyer99.github.io/Recommendation-System-in-Pytorch/)
from typing import Tuple

import pandas as pd
from pytorch_lightning.loggers import mlflow
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from sklearn.preprocessing import MinMaxScaler
from src.dataset import MovielensDataset
from src.model import LightningCollaborativeFiltering

DATA_PATH = "./data/processed/ratings.parquet"

SEED = 42
TRAIN_VAL_RATIO = 0.8

BATCH_SIZE = 256
EMBEDDING_DIM = 20
LEARNING_RATE = 1e-3
DROPOUT = 0.2
MAX_EPOCHS = 50


def process_data(train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(train_data[["Rating"]])

    train_data["ScaledRating"] = scaler.transform(train_data[["Rating"]]).flatten()
    val_data["ScaledRating"] = scaler.transform(val_data[["Rating"]]).flatten()

    return train_data, val_data


def train_val_temporal_split(data, train_val_ratio):
    """Split the `data` in train/validation by time based on `train_val_ratio`.

    This is made by computing the `train_val_ratio` quantile of the "Timestamp"
    column, then splitting accordingly."""
    q_timestamp = data["Timestamp"].quantile(train_val_ratio)
    train_idx = data["Timestamp"] < q_timestamp
    val_idx = data["Timestamp"] >= q_timestamp

    train_ratings = data[train_idx].copy()
    val_ratings = data[val_idx].copy()

    return train_ratings, val_ratings


def main():
    # Define seed
    pl.seed_everything(SEED)

    # Load Data
    ratings = pd.read_parquet(DATA_PATH)

    # Train/Val split
    train_ratings, val_ratings = train_val_temporal_split(ratings, TRAIN_VAL_RATIO)

    # Process data
    train_ratings, val_ratings = process_data(train_ratings, val_ratings)

    # Datasets
    train_dataset = MovielensDataset(data=train_ratings)
    val_dataset = MovielensDataset(data=val_ratings)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = LightningCollaborativeFiltering(num_users=train_dataset.num_users,
                                            num_movies=train_dataset.num_movies,
                                            embedding_dim=EMBEDDING_DIM,
                                            dropout=DROPOUT,
                                            lr=LEARNING_RATE,
                                            batch_size=BATCH_SIZE)
    # Callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor="Loss/Val", min_delta=1e-3, patience=3)

    # Logger
    logger = pl.loggers.MLFlowLogger()

    # Train the model
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, gpus=1, callbacks=[early_stopping], logger=logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
