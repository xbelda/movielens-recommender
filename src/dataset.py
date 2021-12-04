from typing import Dict, Optional, List

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader


class MovielensDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.user_ids = data["UserID"].values
        self.movie_ids = data["MovieID"].values
        self.ratings = data["ScaledRating"].values
        self.genders = data["GenderEncoded"].values
        self.age = data["AgeEncoded"].values
        self.occupations = data["OccupationEncoded"].values
        self.genres = data["GenresList"].values
        self.zip_area = data["Zip-code-Area"].values
        self.zip_section = data["Zip-code-Section"].values

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, int]:
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]
        rating = self.ratings[idx]
        user_gender = self.genders[idx]
        user_age = self.age[idx]
        user_occupation = self.occupations[idx]
        user_zip_area = self.zip_area[idx]
        user_zip_section = self.zip_section[idx]
        movie_genre = self.genres[idx]

        return {"user": user_id,
                "movie": movie_id,
                "rating": rating,
                "user_age": user_age,
                "user_gender": user_gender,
                "user_occupation": user_occupation,
                "user_zip_area": user_zip_area,
                "user_zip_section": user_zip_section,
                "movie_categories": movie_genre}


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
        ratings = pd.read_parquet(self.ratings_dir)
        movies = pd.read_parquet(self.movies_dir)
        users = pd.read_parquet(self.users_dir)

        data = (ratings
                .merge(users, on="UserID", how="left")
                .merge(movies, on="MovieID", how="left"))

        # Scale Rating
        scaler = MinMaxScaler().fit(data[["Rating"]])  # Transform ranges 1 to 5
        data["ScaledRating"] = scaler.transform(data[["Rating"]])

        # Gender
        gender_encoder = LabelEncoder().fit(data["Gender"])
        data["GenderEncoded"] = gender_encoder.transform(data["Gender"])

        # Age
        age_encoder = LabelEncoder().fit(data["Age"])
        data["AgeEncoded"] = age_encoder.transform(data["Age"])

        # Ocupation
        occupation_encoder = LabelEncoder().fit(data["Occupation"])
        data["OccupationEncoded"] = occupation_encoder.transform(data["Occupation"])

        # Genres
        genres = data["Genres"].str.split("|").explode().to_frame()

        genres_encoder = LabelEncoder().fit(genres["Genres"])
        genres["EncodedGenres"] = genres_encoder.transform(genres["Genres"])

        data["GenresList"] = genres.groupby(genres.index)["EncodedGenres"].apply(list)

        # Zip code
        # Note: Zip codes can be divided into regions: https://www.loqate.com/resources/blog/what-is-a-zip-code/
        data["Zip-code-Area"] = data["Zip-code"].str[0].astype(int)
        data["Zip-code-Section"] = data["Zip-code"].str[1:3].astype(int)

        # Save data
        self.data = data

        # Store dimensions
        self.num_users = self.data["UserID"].max() + 1
        self.num_movies = self.data["MovieID"].max() + 1
        self.num_genders = len(gender_encoder.classes_)
        self.num_ages = len(age_encoder.classes_)
        self.num_categories = len(genres_encoder.classes_)
        self.num_occupations = len(occupation_encoder.classes_)

        self.num_zip_areas = 10  # These might not reach 0-9 but this way we can be cautious
        self.num_zip_section = 100

    def setup(self, stage: Optional[str] = None) -> None:
        # Train/Val split
        train_data, val_data = self._train_val_temporal_split(self.data, self.train_test_ratio)

        # Datasets
        self.train_dataset = MovielensDataset(data=train_data)
        self.val_dataset = MovielensDataset(data=val_data)

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
