"""DSSM model implemented with TensorFlow.

This module only contains the DSSM model definition (two-tower architecture).
"""

from __future__ import annotations

import tensorflow as tf


class Tower(tf.keras.layers.Layer):
    """A simple MLP tower used by DSSM."""

    def __init__(self, hidden_dims: list[int], dropout: float = 0.0, name: str | None = None):
        super().__init__(name=name)
        self.blocks = []
        for dim in hidden_dims:
            self.blocks.append(tf.keras.layers.Dense(dim, activation="relu"))
            if dropout > 0:
                self.blocks.append(tf.keras.layers.Dropout(dropout))

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for layer in self.blocks:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


class DSSM(tf.keras.Model):
    """Deep Structured Semantic Model (DSSM) for user-item matching.

    Args:
        num_users: Number of distinct users.
        num_items: Number of distinct items.
        embedding_dim: Embedding dimension for ID embeddings.
        tower_dims: Hidden layer dimensions for both user/item towers.
        dropout: Dropout rate in towers.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        tower_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        tower_dims = tower_dims or [128, 64]

        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.user_tower = Tower(tower_dims, dropout=dropout, name="user_tower")
        self.item_tower = Tower(tower_dims, dropout=dropout, name="item_tower")

    def encode_user(self, user_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        user_vec = self.user_embedding(user_ids)
        user_vec = self.user_tower(user_vec, training=training)
        user_vec = tf.math.l2_normalize(user_vec, axis=-1)
        return user_vec

    def encode_item(self, item_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        item_vec = self.item_embedding(item_ids)
        item_vec = self.item_tower(item_vec, training=training)
        item_vec = tf.math.l2_normalize(item_vec, axis=-1)
        return item_vec

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Returns user-item cosine-like score via dot product."""
        user_vec = self.encode_user(inputs["user_id"], training=training)
        item_vec = self.encode_item(inputs["item_id"], training=training)
        return tf.reduce_sum(user_vec * item_vec, axis=-1)
