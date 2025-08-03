import tensorflow as tf


def dcg_at_k(labels, k):
    """Computes DCG@k for binary relevance."""
    labels = tf.cast(labels, tf.float32)
    gains = (2 ** labels - 1)[:, :k]
    discounts = tf.math.log(tf.range(2, k + 2, dtype=tf.float32))
    return tf.reduce_sum(gains / discounts, axis=1)


def ndcg_at_k(true_items, predicted_items, k):
    """
    Computes nDCG@k.
    true_items: tf.Tensor of shape [batch_size], ground truth indices (0 <= idx < slate_size)
    predicted_items: tf.Tensor of shape [batch_size, slate_size], relevance matrix (e.g., binary/multi-level)
    k: int
    """
    relevant = tf.cast(tf.equal(predicted_items, tf.expand_dims(true_items, axis=1)), tf.float32)
    dcg = dcg_at_k(relevant, k)

    idcg = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    ndcg = dcg / idcg
    return tf.reduce_mean(ndcg)


def slate_mrr(true_items, predicted_items, k):
    """
    Computes MRR@k.
    true_items: tf.Tensor of shape [batch_size]
    predicted_items: tf.Tensor of shape [batch_size, slate_size]
    k: int
    """
    hits = tf.cast(tf.equal(predicted_items[:, :k], tf.expand_dims(true_items, axis=1)), tf.float32)
    has_hit = tf.reduce_max(hits, axis=1)

    first_hit_pos = tf.argmax(hits, axis=1, output_type=tf.int32)
    reciprocal_ranks = tf.where(
        tf.equal(has_hit, 1.0),
        1.0 / (tf.cast(first_hit_pos, tf.float32) + 1.0),
        tf.zeros_like(first_hit_pos, dtype=tf.float32)
    )

    return tf.reduce_mean(reciprocal_ranks)
