"""Train DSSM on MovieLens with multiple training modes.

Supports:
1) pointwise + in-batch negatives
2) pointwise + full negatives sampling
3) pairwise + in-batch negatives
4) pairwise + full negatives sampling
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from dssm_model import DSSM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_movielens_loo(seed: int = 42):
    """Load MovieLens 100k and produce leave-one-out split by user."""
    ds = tfds.load("movielens/100k-ratings", split="train", as_supervised=False)
    records = []
    for ex in tfds.as_numpy(ds):
        user = ex["user_id"].decode("utf-8")
        item = ex["movie_title"].decode("utf-8")
        ts = int(ex["timestamp"])
        records.append((user, item, ts))

    user_hist = defaultdict(list)
    for u, i, ts in records:
        user_hist[u].append((ts, i))

    train_pairs, test_pairs = [], []
    for u, seq in user_hist.items():
        seq.sort(key=lambda x: x[0])
        if len(seq) == 1:
            train_pairs.append((u, seq[0][1]))
            test_pairs.append((u, seq[0][1]))
            continue
        for _, it in seq[:-1]:
            train_pairs.append((u, it))
        test_pairs.append((u, seq[-1][1]))

    users = sorted({u for u, _ in train_pairs} | {u for u, _ in test_pairs})
    items = sorted({i for _, i in train_pairs} | {i for _, i in test_pairs})
    user2idx = {u: idx for idx, u in enumerate(users)}
    item2idx = {i: idx for idx, i in enumerate(items)}

    train_idx = np.array([(user2idx[u], item2idx[i]) for u, i in train_pairs], dtype=np.int32)
    test_idx = np.array([(user2idx[u], item2idx[i]) for u, i in test_pairs], dtype=np.int32)

    rng = np.random.default_rng(seed)
    rng.shuffle(train_idx)

    train_user_items = defaultdict(set)
    for u, i in train_idx:
        train_user_items[int(u)].add(int(i))

    return train_idx, test_idx, user2idx, item2idx, train_user_items


def build_dataset(train_idx: np.ndarray, batch_size: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices({"user_id": train_idx[:, 0], "item_id": train_idx[:, 1]})
    ds = ds.shuffle(min(len(train_idx), 100000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def pointwise_inbatch_loss(user_vec: tf.Tensor, item_vec: tf.Tensor) -> tf.Tensor:
    logits = tf.matmul(user_vec, item_vec, transpose_b=True)  # [B, B]
    labels = tf.eye(tf.shape(logits)[0])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def pairwise_inbatch_loss(user_vec: tf.Tensor, item_vec: tf.Tensor) -> tf.Tensor:
    sim = tf.matmul(user_vec, item_vec, transpose_b=True)  # [B, B]
    pos = tf.linalg.diag_part(sim)[:, None]
    mask = 1.0 - tf.eye(tf.shape(sim)[0])
    diffs = (pos - sim) * mask
    # BPR: -log(sigmoid(pos-neg)) -> softplus(-(pos-neg))
    loss = tf.nn.softplus(-diffs) * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


def sample_negatives_excluding(positives: np.ndarray, num_items: int, num_neg: int) -> np.ndarray:
    negs = np.random.randint(0, num_items, size=(len(positives), num_neg), dtype=np.int32)
    for r in range(len(positives)):
        for c in range(num_neg):
            if negs[r, c] == positives[r]:
                negs[r, c] = (negs[r, c] + 1) % num_items
    return negs


def pointwise_full_loss(model: DSSM, user_ids: tf.Tensor, pos_item_ids: tf.Tensor, num_items: int, num_neg: int, training: bool) -> tf.Tensor:
    user_vec = model.encode_user(user_ids, training=training)
    pos_vec = model.encode_item(pos_item_ids, training=training)
    pos_logits = tf.reduce_sum(user_vec * pos_vec, axis=-1)

    neg_np = sample_negatives_excluding(pos_item_ids.numpy(), num_items=num_items, num_neg=num_neg)
    neg_ids = tf.convert_to_tensor(neg_np)
    neg_vec = model.encode_item(neg_ids, training=training)  # [B, K, D]
    user_exp = tf.expand_dims(user_vec, 1)
    neg_logits = tf.reduce_sum(user_exp * neg_vec, axis=-1)

    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_logits), logits=pos_logits)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_logits), logits=neg_logits)
    return tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)


def pairwise_full_loss(model: DSSM, user_ids: tf.Tensor, pos_item_ids: tf.Tensor, num_items: int, num_neg: int, training: bool) -> tf.Tensor:
    user_vec = model.encode_user(user_ids, training=training)
    pos_vec = model.encode_item(pos_item_ids, training=training)
    pos_logits = tf.reduce_sum(user_vec * pos_vec, axis=-1, keepdims=True)

    neg_np = sample_negatives_excluding(pos_item_ids.numpy(), num_items=num_items, num_neg=num_neg)
    neg_ids = tf.convert_to_tensor(neg_np)
    neg_vec = model.encode_item(neg_ids, training=training)
    user_exp = tf.expand_dims(user_vec, 1)
    neg_logits = tf.reduce_sum(user_exp * neg_vec, axis=-1)

    loss = tf.nn.softplus(-(pos_logits - neg_logits))
    return tf.reduce_mean(loss)


def compute_item_matrix(model: DSSM, num_items: int, batch_size: int = 1024) -> np.ndarray:
    mats = []
    for s in range(0, num_items, batch_size):
        ids = tf.range(s, min(s + batch_size, num_items), dtype=tf.int32)
        mats.append(model.encode_item(ids, training=False).numpy())
    return np.concatenate(mats, axis=0)


def evaluate(model: DSSM, test_idx: np.ndarray, train_user_items: dict[int, set[int]], num_items: int, ks=(5, 10, 20)):
    item_mat = compute_item_matrix(model, num_items)
    user_ids = tf.convert_to_tensor(test_idx[:, 0], dtype=tf.int32)
    true_items = test_idx[:, 1]
    user_mat = model.encode_user(user_ids, training=False).numpy()

    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    mrrs = {k: [] for k in ks}
    aucs = []

    for idx, (u, true_i) in enumerate(test_idx):
        scores = user_mat[idx] @ item_mat.T
        seen = train_user_items.get(int(u), set())
        if seen:
            scores[list(seen)] = -1e9
        rank = int(np.sum(scores > scores[true_i]) + 1)
        auc = float(np.mean(scores[true_i] > np.delete(scores, true_i)))
        aucs.append(auc)

        for k in ks:
            hit = 1.0 if rank <= k else 0.0
            recalls[k].append(hit)
            if rank <= k:
                ndcgs[k].append(1.0 / math.log2(rank + 1))
                mrrs[k].append(1.0 / rank)
            else:
                ndcgs[k].append(0.0)
                mrrs[k].append(0.0)

    metrics = {"AUC": float(np.mean(aucs))}
    for k in ks:
        metrics[f"Recall@{k}"] = float(np.mean(recalls[k]))
        metrics[f"NDCG@{k}"] = float(np.mean(ndcgs[k]))
        metrics[f"MRR@{k}"] = float(np.mean(mrrs[k]))
    return metrics


def train_one_mode(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_user_items: dict[int, set[int]],
    num_users: int,
    num_items: int,
    loss_mode: str,
    neg_mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_neg: int,
    embedding_dim: int,
    tower_dims: list[int],
):
    model = DSSM(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim, tower_dims=tower_dims, dropout=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ds = build_dataset(train_idx, batch_size)

    for ep in range(1, epochs + 1):
        losses = []
        for batch in ds:
            u = tf.cast(batch["user_id"], tf.int32)
            i = tf.cast(batch["item_id"], tf.int32)
            with tf.GradientTape() as tape:
                if neg_mode == "inbatch":
                    user_vec = model.encode_user(u, training=True)
                    item_vec = model.encode_item(i, training=True)
                    if loss_mode == "pointwise":
                        loss = pointwise_inbatch_loss(user_vec, item_vec)
                    else:
                        loss = pairwise_inbatch_loss(user_vec, item_vec)
                else:
                    if loss_mode == "pointwise":
                        loss = pointwise_full_loss(model, u, i, num_items, num_neg=num_neg, training=True)
                    else:
                        loss = pairwise_full_loss(model, u, i, num_items, num_neg=num_neg, training=True)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(float(loss.numpy()))

        print(f"Epoch {ep}/{epochs} - {loss_mode}/{neg_mode} loss: {np.mean(losses):.6f}")

    metrics = evaluate(model, test_idx, train_user_items, num_items)
    return model, metrics


def run_experiments(args):
    train_idx, test_idx, user2idx, item2idx, train_user_items = load_movielens_loo(seed=args.seed)
    num_users = len(user2idx)
    num_items = len(item2idx)

    modes = []
    if args.loss_mode == "all":
        loss_modes = ["pointwise", "pairwise"]
    else:
        loss_modes = [args.loss_mode]

    if args.negative_sampling == "all":
        neg_modes = ["inbatch", "full"]
    else:
        neg_modes = [args.negative_sampling]

    for l in loss_modes:
        for n in neg_modes:
            modes.append((l, n))

    print(f"Train interactions: {len(train_idx)}, Test users: {len(test_idx)}, Users: {num_users}, Items: {num_items}")

    results = []
    for l, n in modes:
        print("\n" + "=" * 60)
        print(f"Training mode: loss={l}, negatives={n}")
        _, metrics = train_one_mode(
            train_idx=train_idx,
            test_idx=test_idx,
            train_user_items=train_user_items,
            num_users=num_users,
            num_items=num_items,
            loss_mode=l,
            neg_mode=n,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_neg=args.num_neg,
            embedding_dim=args.embedding_dim,
            tower_dims=args.tower_dims,
        )
        results.append((l, n, metrics))
        print(f"Metrics ({l}/{n}): {metrics}")

    print("\n" + "#" * 80)
    print("Final comparison across training modes")
    header = ["loss_mode", "neg_mode", "AUC", "Recall@5", "Recall@10", "NDCG@10", "MRR@10"]
    print("\t".join(header))
    for l, n, m in results:
        row = [
            l,
            n,
            f"{m['AUC']:.4f}",
            f"{m['Recall@5']:.4f}",
            f"{m['Recall@10']:.4f}",
            f"{m['NDCG@10']:.4f}",
            f"{m['MRR@10']:.4f}",
        ]
        print("\t".join(row))


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSSM on MovieLens with multiple training schemes")
    parser.add_argument("--loss_mode", type=str, default="all", choices=["pointwise", "pairwise", "all"])
    parser.add_argument("--negative_sampling", type=str, default="all", choices=["inbatch", "full", "all"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_neg", type=int, default=10, help="number of random negatives for full sampling")
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--tower_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    run_experiments(args)
