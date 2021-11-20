# Bibliography
# - [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)
# - [Getting Started with a Movie Recommendation System](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system#Content-Based-Filtering)
# - [Collaborative Filtering in Pytorch](https://spiyer99.github.io/Recommendation-System-in-Pytorch/)
import hydra
from omegaconf import OmegaConf
from pytorch_lightning.loggers import mlflow
import pytorch_lightning as pl

from src.dataset import MovielensDataModule
from src.model import LightningCollaborativeFiltering


def main() -> None:
    # Load configuration
    cfg = OmegaConf.load("conf/config.yaml")

    # Define seed
    pl.seed_everything(cfg.SEED)

    # Load Data
    datamodule = MovielensDataModule(cfg.DATA_PATH,
                                     batch_size=cfg.BATCH_SIZE,
                                     train_val_ratio=cfg.TRAIN_VAL_RATIO)

    # Compute internal parameters
    datamodule.prepare_data()

    # Model
    model = LightningCollaborativeFiltering(num_users=datamodule.num_users,
                                            num_movies=datamodule.num_movies,
                                            embedding_dim=cfg.EMBEDDING_DIM,
                                            dropout=cfg.DROPOUT,
                                            lr=cfg.LEARNING_RATE)
    # Callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor="Loss/Val", min_delta=1e-3, patience=3)

    # Logger
    logger = pl.loggers.MLFlowLogger()

    # Train the model
    trainer = pl.Trainer(max_epochs=cfg.MAX_EPOCHS, gpus=1,
                         callbacks=[early_stopping], logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
