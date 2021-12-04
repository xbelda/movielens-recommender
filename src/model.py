from typing import Tuple

import torch
import pytorch_lightning as pl


class UserEncoder(torch.nn.Module):
    def __init__(self,
                 num_users: int,
                 num_ages: int,
                 num_genders: int,
                 num_occupations: int,
                 num_zip_areas: int,
                 common_embedding_dim: int,
                 user_embedding_dim: int,
                 user_age_embedding_dim: int,
                 user_gender_embedding_dim: int,
                 user_occupation_embedding_dim: int,
                 user_zip_area_embedding_dim: int,
                 dropout: float):
        super().__init__()

        self.emb_users = torch.nn.Embedding(num_embeddings=num_users,
                                            embedding_dim=user_embedding_dim)

        self.bias_user = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=1)

        self.emb_age = torch.nn.Embedding(num_embeddings=num_ages,
                                          embedding_dim=user_age_embedding_dim)

        self.emb_gender = torch.nn.Embedding(num_embeddings=num_genders,
                                             embedding_dim=user_gender_embedding_dim)

        self.emb_occupation = torch.nn.Embedding(num_embeddings=num_occupations,
                                                 embedding_dim=user_occupation_embedding_dim)

        self.emb_zip_area = torch.nn.Embedding(num_embeddings=num_zip_areas,
                                               embedding_dim=user_zip_area_embedding_dim)

        fc_input_dim = (user_embedding_dim
                        + user_age_embedding_dim
                        + user_gender_embedding_dim
                        + user_occupation_embedding_dim
                        + user_zip_area_embedding_dim)
        self.fc = torch.nn.Linear(in_features=fc_input_dim,
                                  out_features=common_embedding_dim)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,
                user_id: torch.Tensor,
                age: torch.Tensor,
                gender: torch.Tensor,
                occupation: torch.Tensor,
                zip_area: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_user_vec = self.emb_users(user_id)
        user_age = self.emb_age(age)
        user_gender = self.emb_gender(gender)
        user_occupation = self.emb_occupation(occupation)
        user_zip_area = self.emb_zip_area(zip_area)

        user_vec = torch.cat([raw_user_vec, user_age, user_gender, user_occupation, user_zip_area], dim=1)
        user_vec = self.dropout(self.relu(user_vec))
        user_vec = self.fc(user_vec)

        user_bias = self.bias_user(user_id).flatten()
        return user_vec, user_bias


class MovieEncoder(torch.nn.Module):
    def __init__(self,
                 num_movies: int,
                 num_categories: int,
                 common_embedding_dim: int,
                 movie_embedding_dim: int,
                 movie_category_dim: int,
                 dropout: float
                 ):
        super().__init__()

        self.emb_movies = torch.nn.Embedding(num_embeddings=num_movies,
                                             embedding_dim=movie_embedding_dim)

        self.emb_movie_cats = torch.nn.EmbeddingBag(num_embeddings=num_categories,
                                                    embedding_dim=movie_category_dim,
                                                    mode="mean", padding_idx=0)
        self.bias_movie = torch.nn.Embedding(num_embeddings=num_movies, embedding_dim=1)

        self.fc = torch.nn.Linear(in_features=movie_embedding_dim + movie_category_dim,
                                  out_features=common_embedding_dim)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, movie_id, movie_categories):
        raw_movie_vec = self.emb_movies(movie_id)
        categories_vec = self.emb_movie_cats(movie_categories)

        movie_vec = torch.cat([raw_movie_vec, categories_vec], dim=1)
        movie_vec = self.dropout(self.relu(movie_vec))
        movie_vec = self.fc(movie_vec)

        movie_bias = self.bias_movie(movie_id).flatten()
        return movie_vec, movie_bias


class LightningCollaborativeFiltering(pl.LightningModule):
    def __init__(self, num_users: int,
                 num_movies: int,
                 num_categories: int,
                 num_ages: int,
                 num_genders: int,
                 num_occupations: int,
                 num_zip_areas: int,
                 common_embedding_dim: int,
                 user_embedding_dim: int,
                 movie_embedding_dim: int,
                 movie_category_dim: int,
                 user_age_embedding_dim: int,
                 user_gender_embedding_dim: int,
                 user_occupation_embedding_dim: int,
                 user_zip_area_embedding_dim: int,
                 dropout: float,
                 lr: float = 1e-2,
                 margin: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.margin = margin
        self.loss = torch.nn.MSELoss()

        self.user_encoder = UserEncoder(num_users=num_users,
                                        num_ages=num_ages,
                                        num_genders=num_genders,
                                        num_occupations=num_occupations,
                                        num_zip_areas=num_zip_areas,
                                        common_embedding_dim=common_embedding_dim,
                                        user_embedding_dim=user_embedding_dim,
                                        user_age_embedding_dim=user_age_embedding_dim,
                                        user_gender_embedding_dim=user_gender_embedding_dim,
                                        user_occupation_embedding_dim=user_occupation_embedding_dim,
                                        user_zip_area_embedding_dim=user_zip_area_embedding_dim,
                                        dropout=dropout)

        self.movie_encoder = MovieEncoder(num_movies=num_movies,
                                          num_categories=num_categories,
                                          common_embedding_dim=common_embedding_dim,
                                          movie_embedding_dim=movie_embedding_dim,
                                          movie_category_dim=movie_category_dim,
                                          dropout=dropout)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, users, user_age, user_gender, user_occupation, user_zip_area, movies, movie_categories):
        user_vec, user_bias = self.user_encoder(users, user_age, user_gender, user_occupation, user_zip_area)
        movie_vec, movie_bias = self.movie_encoder(movies, movie_categories)

        product = (user_vec * movie_vec).sum(dim=1)

        # Add bias
        scaled_product = product + user_bias + movie_bias

        out = torch.sigmoid(scaled_product)
        # Rescale output (sigmoid saturates at 0 and 1)
        out = out * (1.0 + 2 * self.margin) - self.margin

        return out

    def _base_step(self, batch, batch_idx):
        users = batch["user"]
        movies = batch["movie"]
        ratings = batch["rating"]

        user_age = batch["user_age"]
        user_gender = batch["user_gender"]
        user_occupation = batch["user_occupation"]
        user_zip_area = batch["user_zip_area"]

        movie_categories = batch["movie_categories"]

        ratings_pred = self(users, user_age, user_gender, user_occupation, user_zip_area, movies, movie_categories)

        # Convert to same dtypes
        ratings = ratings.to(ratings_pred.dtype)

        loss = self.loss(ratings_pred, ratings)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._base_step(batch, batch_idx)
        self.log('Loss/Train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._base_step(batch, batch_idx)
        self.log('Loss/Val', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._base_step(batch, batch_idx)
        self.log('Loss/Test', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
