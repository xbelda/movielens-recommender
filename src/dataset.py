from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from src.encoding import LabelEncoder


class MovielensDataset(Dataset):
    def __init__(self,
                 scores: pd.DataFrame,
                 movies: pd.DataFrame,
                 users: pd.DataFrame,
                 user_encoder: LabelEncoder,
                 movie_encoder: LabelEncoder,
                 movie_categories_encoder: LabelEncoder,
                 user_gender_encoder: LabelEncoder,
                 user_age_encoder: LabelEncoder):
        self.user_ids = scores["UserID"].values
        self.movie_ids = scores["MovieID"].values
        self.ratings = scores["ScaledRating"].values

        self.user_info = users.set_index("UserID")

        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder

        self.movie_categories = movies.set_index("MovieID")["Genres"]
        self.movie_categories_encoder = movie_categories_encoder

        self.user_gender_encoder = user_gender_encoder
        self.user_age_encoder = user_age_encoder

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, int]:
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]
        rating = self.ratings[idx]

        # Transform IDs
        new_user_id = self.user_encoder.transform(user_id)
        new_movie_id = self.movie_encoder.transform(movie_id)

        # Encode categories
        raw_categories = self.movie_categories.loc[movie_id]
        categories = raw_categories.split("|")
        encoded_categories = [self.movie_categories_encoder.transform(c) for c in categories]

        # Encode User data
        user_data = self.user_info.loc[user_id]

        user_age = self.user_age_encoder.transform(user_data["Age"])
        user_gender = self.user_gender_encoder.transform(user_data["Gender"])

        return {"user": new_user_id,
                "movie": new_movie_id,
                "rating": rating,
                "movie_categories": encoded_categories,
                "user_age": user_age,
                "user_gender": user_gender}


class MovielensDataModule(pl.LightningDataModule):
    def __init__(self,
                 ratings_dir: str,
                 movies_dir: str,
                 users_dir: str,
                 batch_size: int,
                 train_val_ratio: float):
        super().__init__()
        self.ratings_dir = ratings_dir
        self.movies_dir = movies_dir
        self.users_dir = users_dir
        self.batch_size = batch_size
        self.train_test_ratio = train_val_ratio

    def prepare_data(self):
        # Load Data
        self.ratings = pd.read_parquet(self.ratings_dir)
        self.movies = pd.read_parquet(self.movies_dir)
        self.users = pd.read_parquet(self.users_dir)

        # Generate encoders
        self.user_encoder = LabelEncoder().fit(self.ratings["UserID"].unique())
        self.movie_encoder = LabelEncoder().fit(self.ratings["MovieID"].unique())

        unique_movie_cats = self.movies["Genres"].str.split("|").explode().unique()
        self.movie_cat_encoder = LabelEncoder().fit(unique_movie_cats)

        self.user_gender_encoder = LabelEncoder(handle_unknown=False).fit(self.users["Gender"].unique())
        self.user_age_encoder = LabelEncoder(handle_unknown=False).fit(self.users["Age"].unique())

        # Scaling
        self.scaler = MinMaxScaler().fit([[1], [5]])  # Transform ranges 1 to 5

    def _base_setup(self, ratings: pd.DataFrame) -> MovielensDataset:
        ratings["ScaledRating"] = self.scaler.transform(ratings[["Rating"]]).flatten()

        # Datasets
        dataset = MovielensDataset(scores=ratings,
                                   movies=self.movies,
                                   users=self.users,
                                   user_encoder=self.user_encoder,
                                   movie_encoder=self.movie_encoder,
                                   movie_categories_encoder=self.movie_cat_encoder,
                                   user_age_encoder=self.user_age_encoder,
                                   user_gender_encoder=self.user_gender_encoder)
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        # Train/Val split
        train_ratings, val_ratings = self._train_val_temporal_split(self.ratings, self.train_test_ratio)

        # Datasets
        self.train_dataset = self._base_setup(train_ratings)
        self.val_dataset = self._base_setup(val_ratings)

    @staticmethod
    def _collate_fn(batch: List):
        """Merges different examples of the dataset into a single batch.

        Note: This custom is necessary because the default pytorch collate_fn
        tries to batch all `categories` into a single batch and fails due to
        each movie having a different number of categories.
        """

        padding_keys = {"movie_categories"}

        batch_keys = batch[0].keys()

        new_batch = dict()
        for k in batch_keys:
            if k in padding_keys:
                value = [torch.tensor(example[k]) for example in batch]
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            else:
                value = torch.tensor([example[k] for example in batch])
            new_batch[k] = value

        return new_batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            collate_fn=self._collate_fn
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
