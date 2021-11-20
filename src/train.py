# Bibliography
# - [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)
# - [Getting Started with a Movie Recommendation System](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system#Content-Based-Filtering)
# - [Collaborative Filtering in Pytorch](https://spiyer99.github.io/Recommendation-System-in-Pytorch/)
from typing import Tuple

import pandas as pd
from pytorch_lightning.loggers import mlflow
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import optuna

from sklearn.preprocessing import MinMaxScaler
from src.dataset import MovielensDataset
from src.model import LightningCollaborativeFiltering

DATA_PATH = "./data/processed/ratings.parquet"

SEED = 42
TRAIN_VAL_RATIO = 0.8
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


def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    batch_size = trial.suggest_int("batch_size", 32, 256, log=True)
    embedding_dim = trial.suggest_int("embedding_dim", 8, 32, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

    hyperparameters = dict(batch_size=batch_size,
                           embedding_dim=embedding_dim,
                           dropout=dropout,
                           lr=lr,
                           seed=SEED)

    # Set seed
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = LightningCollaborativeFiltering(num_users=train_dataset.num_users,
                                            num_movies=train_dataset.num_movies,
                                            embedding_dim=embedding_dim,
                                            dropout=dropout,
                                            lr=lr)
    # Callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor="Loss/Val",
                                                min_delta=1e-3,
                                                patience=3)

    # Logger
    logger = pl.loggers.MLFlowLogger(experiment_name="hyperparameter_tuning",
                                     run_name=f"trial_{trial.number}")

    # Train the model
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, gpus=1, callbacks=[early_stopping], logger=logger)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return trainer.callback_metrics["Loss/Val"].item()


if __name__ == "__main__":
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=20, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
