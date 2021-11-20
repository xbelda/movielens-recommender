# Bibliography
# - [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)
# - [Getting Started with a Movie Recommendation System](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system#Content-Based-Filtering)
# - [Collaborative Filtering in Pytorch](https://spiyer99.github.io/Recommendation-System-in-Pytorch/)
from typing import Tuple

from pytorch_lightning.loggers import mlflow
import pytorch_lightning as pl

from src.dataset import MovielensDataModule
from src.model import LightningCollaborativeFiltering

DATA_PATH = "./data/processed/ratings.parquet"

SEED = 42
TRAIN_VAL_RATIO = 0.8

BATCH_SIZE = 256
EMBEDDING_DIM = 20
LEARNING_RATE = 1e-3
DROPOUT = 0.2
MAX_EPOCHS = 50


def main():
    # Define seed
    pl.seed_everything(SEED)

    # Load Data
    datamodule = MovielensDataModule(DATA_PATH,
                                     batch_size=BATCH_SIZE,
                                     train_val_ratio=TRAIN_VAL_RATIO)

    # Compute internal parameters
    datamodule.prepare_data()

    # Model
    model = LightningCollaborativeFiltering(num_users=datamodule.num_users,
                                            num_movies=datamodule.num_movies,
                                            embedding_dim=EMBEDDING_DIM,
                                            dropout=DROPOUT,
                                            lr=LEARNING_RATE)
    # Callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor="Loss/Val", min_delta=1e-3, patience=3)

    # Logger
    logger = pl.loggers.MLFlowLogger()

    # Train the model
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, gpus=1,
                         callbacks=[early_stopping], logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
