from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


class MovielensDataset(Dataset):
    def __init__(self, users: np.array, movies: np.array, ratings: np.array):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, int]:
        return {"user": self.users[idx],
                "movie": self.movies[idx],
                "rating": self.ratings[idx]}


class MovielensDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, train_val_ratio: float):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_test_ratio = train_val_ratio

    def prepare_data(self):
        data = pd.read_parquet(self.data_dir)

        # Convert old to new ids
        u2id = {u_id: new_u_id for new_u_id, u_id in enumerate(data["UserID"].unique())}
        m2id = {m_id: new_m_id for new_m_id, m_id in enumerate(data["MovieID"].unique())}

        self.num_users = len(u2id)
        self.num_movies = len(m2id)

        data["NewUserID"] = data["UserID"].map(u2id)
        data["NewMovieID"] = data["MovieID"].map(m2id)

        self.data = data

    def setup(self, stage: Optional[str] = None) -> None:
        # Train/Val split
        train_ratings, val_ratings = self._train_val_temporal_split(self.data, self.train_test_ratio)

        # Process data
        train_ratings, val_ratings = self._process_data(train_ratings, val_ratings)

        # Datasets
        self.train_dataset = MovielensDataset(users=train_ratings["NewUserID"].values,
                                              movies=train_ratings["NewMovieID"].values,
                                              ratings=train_ratings["ScaledRating"].values)

        self.val_dataset = MovielensDataset(users=val_ratings["NewUserID"].values,
                                            movies=val_ratings["NewMovieID"].values,
                                            ratings=val_ratings["ScaledRating"].values)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

    @staticmethod
    def _train_val_temporal_split(data, train_val_ratio):
        """Splits `data` in train/validation by time based on `train_val_ratio`.

        This is made by computing the `train_val_ratio` quantile of the "Timestamp"
        column, then splitting accordingly."""
        q_timestamp = data["Timestamp"].quantile(train_val_ratio)
        train_idx = data["Timestamp"] < q_timestamp
        val_idx = data["Timestamp"] >= q_timestamp

        train_ratings = data[train_idx].copy()
        val_ratings = data[val_idx].copy()

        return train_ratings, val_ratings

    @staticmethod
    def _process_data(train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Scaling
        scaler = MinMaxScaler()
        scaler.fit(train_data[["Rating"]])

        train_data["ScaledRating"] = scaler.transform(train_data[["Rating"]]).flatten()
        val_data["ScaledRating"] = scaler.transform(val_data[["Rating"]]).flatten()

        return train_data, val_data
