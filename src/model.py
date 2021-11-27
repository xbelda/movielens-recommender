import torch
import pytorch_lightning as pl


class CollaborativeFiltering(torch.nn.Module):
    def __init__(self,
                 num_users: int,
                 num_movies: int,
                 num_categories: int,
                 user_embedding_dim: int,
                 movie_embedding_dim: int,
                 movie_category_dim: int,
                 dropout=float):
        super().__init__()
        # TODO: Split User and Movie Networks

        self.margin = 0.1

        assert movie_embedding_dim + movie_category_dim == user_embedding_dim, "Movie dims should have the same length as User dims"

        self.emb_users = torch.nn.Embedding(num_embeddings=num_users,
                                            embedding_dim=user_embedding_dim)

        self.emb_movies = torch.nn.Embedding(num_embeddings=num_movies,
                                             embedding_dim=movie_embedding_dim)

        self.emb_movie_cats = torch.nn.EmbeddingBag(num_embeddings=num_categories,
                                                    embedding_dim=movie_category_dim,
                                                    mode="mean", padding_idx=0)

        self.bias_user = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=1)
        self.bias_movie = torch.nn.Embedding(num_embeddings=num_movies, embedding_dim=1)

        self.dropout = torch.nn.Dropout(dropout)

        # TODO:
        #  Add content information
        #   - Number of votes per user/movie
        #   - User profile data

    def forward(self,
                user_id: torch.Tensor,
                movie_id: torch.Tensor,
                movie_categories: torch.Tensor
                ) -> torch.Tensor:
        user_vec = self.emb_users(user_id)
        movie_vec = torch.cat([self.emb_movies(movie_id),
                               self.emb_movie_cats(movie_categories)], dim=1)

        product = (user_vec * movie_vec).sum(dim=1)

        product = self.dropout(product)

        # Add bias
        scaled_product = (product
                          + self.bias_user(user_id).flatten()
                          + self.bias_movie(movie_id).flatten())

        scaled_product = self.dropout(scaled_product)

        out = torch.sigmoid(scaled_product)
        # Rescale output (sigmoid saturates at 0 and 1)
        out = out * (1.0 + 2 * self.margin) - self.margin
        return out


class LightningCollaborativeFiltering(pl.LightningModule):
    def __init__(self, num_users: int,
                 num_movies: int,
                 num_categories: int,
                 user_embedding_dim: int,
                 movie_embedding_dim: int,
                 movie_category_dim: int,
                 dropout=float,
                 lr=1e-2):
        super().__init__()
        self.save_hyperparameters()

        self.model = CollaborativeFiltering(num_users=num_users,
                                            num_movies=num_movies,
                                            num_categories=num_categories,
                                            user_embedding_dim=user_embedding_dim,
                                            movie_embedding_dim=movie_embedding_dim,
                                            movie_category_dim=movie_category_dim,
                                            dropout=dropout)

        self.lr = lr

        self.loss = torch.nn.MSELoss()

    def forward(self, users, movies, movie_categories):
        return self.model(users, movies, movie_categories)

    def _base_step(self, batch, batch_idx):
        users = batch["user"]
        movies = batch["movie"]
        ratings = batch["rating"]
        movie_categories = batch["movie_categories"]

        ratings_pred = self(users, movies, movie_categories)

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
