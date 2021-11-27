from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from src.encoding import Vocabulary


class MovielensDataset(Dataset):
    def __init__(self,
                 scores: pd.DataFrame,
                 user_vocab: Vocabulary,
                 movie_vocab: Vocabulary,
                 movies: pd.DataFrame,
                 movie_categories_vocab: Vocabulary):
        self.user_ids = scores["UserID"].values
        self.movie_ids = scores["MovieID"].values
        self.ratings = scores["ScaledRating"].values

        self.user_vocab = user_vocab
        self.movie_vocab = movie_vocab

        self.movie_categories = movies.set_index("MovieID")["Genres"]
        self.movie_categories_vocab = movie_categories_vocab

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, int]:
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]
        rating = self.ratings[idx]

        # Transform IDs
        new_user_id = self.user_vocab.transform(user_id)
        new_movie_id = self.movie_vocab.transform(movie_id)

        # Encode categories
        raw_categories = self.movie_categories.loc[movie_id]
        categories = raw_categories.split("|")
        encoded_categories = [self.movie_categories_vocab.transform(c) for c in categories]

        return {"user": new_user_id,
                "movie": new_movie_id,
                "rating": rating,
                "movie_categories": encoded_categories}


class MovielensDataModule(pl.LightningDataModule):
    def __init__(self,
                 ratings_dir: str,
                 movies_dir: str,
                 batch_size: int,
                 train_val_ratio: float):
        super().__init__()
        self.ratings_dir = ratings_dir
        self.movies_dir = movies_dir
        self.batch_size = batch_size
        self.train_test_ratio = train_val_ratio

    def prepare_data(self):
        # Load Data
        ratings = pd.read_parquet(self.ratings_dir)
        movies = pd.read_parquet(self.movies_dir)

        self.ratings = ratings
        self.movies = movies

        # Generate vocabs
        self.user_vocab = Vocabulary(handle_unknown=True).fit(ratings["UserID"].unique())
        self.movie_vocab = Vocabulary(handle_unknown=True).fit(ratings["MovieID"].unique())

        unique_movie_cats = movies["Genres"].str.split("|").explode().unique()
        self.movie_cat_vocab = Vocabulary(handle_unknown=True).fit(unique_movie_cats)

        self.scaler = MinMaxScaler().fit([[1], [5]])  # Transform ranges 1 to 5

    def _base_setup(self, ratings: pd.DataFrame) -> MovielensDataset:
        ratings["ScaledRating"] = self.scaler.transform(ratings[["Rating"]]).flatten()

        # Datasets
        dataset = MovielensDataset(scores=ratings,
                                   user_vocab=self.user_vocab,
                                   movie_vocab=self.movie_vocab,
                                   movies=self.movies,
                                   movie_categories_vocab=self.movie_cat_vocab)
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
        new_batch = defaultdict(list)

        padding_keys = {"movie_categories"}

        for example in batch:
            for k, v in example.items():
                new_batch[k].append(v)

        for k, v in new_batch.items():
            if k not in padding_keys:
                new_batch[k] = torch.tensor(v)
            else:
                tensor_list = list(map(torch.tensor, new_batch[k]))
                new_batch[k] = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)

        return new_batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
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
