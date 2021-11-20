from typing import Dict

import pandas as pd
from torch.utils.data import Dataset


class MovielensDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.users = data["UserID"].values
        self.movies = data["MovieID"].values
        self.ratings = data["ScaledRating"].values

        # TODO: Move this pre-processing outside the dataset
        # Convert old to new ids
        self.u2id = {u_id: new_u_id for new_u_id, u_id in enumerate(set(self.users))}
        self.m2id = {m_id: new_m_id for new_m_id, m_id in enumerate(set(self.movies))}

        self.num_users = len(self.u2id)
        self.num_movies = len(self.m2id)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, int]:
        user_id = self.users[idx]
        movie_id = self.movies[idx]
        rating = self.ratings[idx]

        # Transform ids
        new_user_id = self.u2id[user_id]
        new_movie_id = self.m2id[movie_id]

        return {"user": new_user_id,
                "movie": new_movie_id,
                "rating": rating}
