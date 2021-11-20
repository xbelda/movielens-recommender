import torch
import pytorch_lightning as pl


class CollaborativeFiltering(torch.nn.Module):
    def __init__(self, num_users: int, num_movies: int, embedding_dim: int, dropout=float):
        super().__init__()

        self.margin = 0.1

        self.emb_users = torch.nn.Embedding(num_embeddings=num_users,
                                            embedding_dim=embedding_dim)

        self.emb_movies = torch.nn.Embedding(num_embeddings=num_movies,
                                             embedding_dim=embedding_dim)

        self.bias_user = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=1)
        self.bias_movie = torch.nn.Embedding(num_embeddings=num_movies, embedding_dim=1)

        self.dropout = torch.nn.Dropout(dropout)

        # TODO:
        #  Add content information
        #   - Number of votes per user/movie
        #   - User profile data
        #   - Movie genre data

    def forward(self, user_id: torch.Tensor, movie_id: torch.Tensor) -> torch.Tensor:
        product = (self.emb_users(user_id) * self.emb_movies(movie_id)).sum(dim=1)

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
                 embedding_dim: int,
                 dropout=float,
                 lr=1e-2,
                 batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()

        self.model = CollaborativeFiltering(num_users=num_users,
                                            num_movies=num_movies,
                                            embedding_dim=embedding_dim,
                                            dropout=dropout)
        self.lr = lr

        self.loss = torch.nn.MSELoss()

    def forward(self, users, movies):
        return self.model(users, movies)

    def _base_step(self, batch, batch_idx):
        users = batch["user"]
        movies = batch["movie"]
        ratings = batch["rating"]

        ratings_pred = self(users, movies)

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
