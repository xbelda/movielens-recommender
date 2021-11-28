import torch
import pytorch_lightning as pl


class UserEncoder(torch.nn.Module):
    def __init__(self,
                 num_users: int,
                 num_ages: int,
                 num_genders: int,
                 user_embedding_dim: int,
                 user_age_embedding_dim: int,
                 user_gender_embedding_dim: int,
                 ):
        super().__init__()

        self.emb_users = torch.nn.Embedding(num_embeddings=num_users,
                                            embedding_dim=user_embedding_dim)

        self.bias_user = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=1)

        self.emb_age = torch.nn.Embedding(num_embeddings=num_ages,
                                          embedding_dim=user_age_embedding_dim)

        self.emb_gender = torch.nn.Embedding(num_embeddings=num_genders,
                                             embedding_dim=user_gender_embedding_dim)

        self.fc = torch.nn.Linear(in_features=user_embedding_dim + user_age_embedding_dim + user_gender_embedding_dim,
                                  out_features=20)

    def forward(self, user_id, age, gender):
        raw_user_vec = self.emb_users(user_id)
        user_age = self.emb_age(age)
        user_gender = self.emb_gender(gender)

        user_vec = torch.cat([raw_user_vec, user_age, user_gender], dim=1)
        user_vec = self.fc(user_vec)

        user_bias = self.bias_user(user_id).flatten()
        return user_vec, user_bias


class MovieEncoder(torch.nn.Module):
    def __init__(self,
                 num_movies: int,
                 num_categories: int,
                 movie_embedding_dim: int,
                 movie_category_dim: int):
        super().__init__()

        self.emb_movies = torch.nn.Embedding(num_embeddings=num_movies,
                                             embedding_dim=movie_embedding_dim)

        self.emb_movie_cats = torch.nn.EmbeddingBag(num_embeddings=num_categories,
                                                    embedding_dim=movie_category_dim,
                                                    mode="mean", padding_idx=0)
        self.bias_movie = torch.nn.Embedding(num_embeddings=num_movies, embedding_dim=1)

        self.fc = torch.nn.Linear(in_features=movie_embedding_dim + movie_category_dim,
                                  out_features=20)

    def forward(self, movie_id, movie_categories):
        raw_movie_vec = self.emb_movies(movie_id)
        categories_vec = self.emb_movie_cats(movie_categories)

        movie_vec = torch.cat([raw_movie_vec, categories_vec], dim=1)
        movie_vec = self.fc(movie_vec)

        movie_bias = self.bias_movie(movie_id).flatten()
        return movie_vec, movie_bias


class CollaborativeFiltering(torch.nn.Module):
    def __init__(self,
                 num_users: int,
                 num_movies: int,
                 num_categories: int,
                 num_ages: int,
                 num_genders: int,
                 user_embedding_dim: int,
                 movie_embedding_dim: int,
                 movie_category_dim: int,
                 user_age_embedding_dim: int,
                 user_gender_embedding_dim: int,
                 dropout: float):
        super().__init__()

        self.margin = 0.1

        assert movie_embedding_dim + movie_category_dim == user_embedding_dim + user_age_embedding_dim + user_gender_embedding_dim, "Movie dims should have the same length as User dims"

        self.user_encoder = UserEncoder(num_users=num_users,
                                        num_ages=num_ages,
                                        num_genders=num_genders,
                                        user_embedding_dim=user_embedding_dim,
                                        user_age_embedding_dim=user_age_embedding_dim,
                                        user_gender_embedding_dim=user_gender_embedding_dim)

        self.movie_encoder = MovieEncoder(num_movies=num_movies,
                                          num_categories=num_categories,
                                          movie_embedding_dim=movie_embedding_dim,
                                          movie_category_dim=movie_category_dim)

        self.dropout = torch.nn.Dropout(dropout)

        # TODO:
        #  Add content information
        #   - Number of votes per user/movie
        #   - User profile data
        #   - Add ocupation/zip

    def forward(self,
                user_id: torch.Tensor,
                user_age: torch.Tensor,
                user_gender: torch.Tensor,
                movie_id: torch.Tensor,
                movie_categories: torch.Tensor
                ) -> torch.Tensor:
        user_vec, user_bias = self.user_encoder(user_id, user_age, user_gender)
        movie_vec, movie_bias = self.movie_encoder(movie_id, movie_categories)

        user_vec = self.dropout(user_vec)
        movie_vec = self.dropout(movie_vec)

        product = (user_vec * movie_vec).sum(dim=1)

        product = self.dropout(product)

        # Add bias
        scaled_product = product + user_bias + movie_bias

        scaled_product = self.dropout(scaled_product)

        out = torch.sigmoid(scaled_product)
        # Rescale output (sigmoid saturates at 0 and 1)
        out = out * (1.0 + 2 * self.margin) - self.margin
        return out


class LightningCollaborativeFiltering(pl.LightningModule):
    def __init__(self, num_users: int,
                 num_movies: int,
                 num_categories: int,
                 num_ages: int,
                 num_genders: int,
                 user_embedding_dim: int,
                 movie_embedding_dim: int,
                 movie_category_dim: int,
                 user_age_embedding_dim: int,
                 user_gender_embedding_dim: int,
                 dropout: float,
                 lr=1e-2):
        super().__init__()
        self.save_hyperparameters()

        self.model = CollaborativeFiltering(num_users=num_users,
                                            num_movies=num_movies,
                                            num_categories=num_categories,
                                            num_ages=num_ages,
                                            num_genders=num_genders,
                                            user_embedding_dim=user_embedding_dim,
                                            movie_embedding_dim=movie_embedding_dim,
                                            movie_category_dim=movie_category_dim,
                                            user_age_embedding_dim=user_age_embedding_dim,
                                            user_gender_embedding_dim=user_gender_embedding_dim,
                                            dropout=dropout)

        self.lr = lr

        self.loss = torch.nn.MSELoss()

    def forward(self, users, user_age, user_gender, movies, movie_categories):
        return self.model(users, user_age, user_gender, movies, movie_categories)

    def _base_step(self, batch, batch_idx):
        users = batch["user"]
        movies = batch["movie"]
        ratings = batch["rating"]
        movie_categories = batch["movie_categories"]
        user_age = batch["user_age"]
        user_gender = batch["user_gender"]

        ratings_pred = self(users, user_age, user_gender, movies, movie_categories)

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
