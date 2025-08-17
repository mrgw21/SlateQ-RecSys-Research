import gin
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from recsim_ng.entities.state_models import static
from recsim_ng.core import value
from recsim_ng.core.value import ValueSpec, FieldSpec


class TensorFieldSpec(FieldSpec):
    def __init__(self, shape, dtype=tf.float32):
        super().__init__() 
        self._shape = shape
        self._dtype = dtype

    def invariant(self):
        return tf.TensorSpec(shape=self._shape, dtype=self._dtype)


@gin.configurable
class ECommUser(static.StaticStateModel):
    def __init__(self, num_topics=10, num_users=1, beta=5.0, reward_mode="sigmoid"):
        super().__init__()
        self.num_topics = num_topics
        self.num_users  = num_users
        self.beta       = beta             # temperature for choice model
        self.reward_mode = reward_mode     # "sigmoid" | "clip01" | "raw"

    def specs(self):
        return ValueSpec(
            interest=TensorFieldSpec(shape=(self.num_users, self.num_topics), dtype=tf.float32),
            choice  =TensorFieldSpec(shape=(self.num_users,), dtype=tf.int32),
            reward  =TensorFieldSpec(shape=(self.num_users,), dtype=tf.float32)
        )

    def initial_state(self):
        interest = tfd.Normal(loc=0., scale=1.).sample((self.num_users, self.num_topics))
        return value.Value(
            interest=interest,
            choice=tf.zeros([self.num_users], dtype=tf.int32),
            reward=tf.zeros([self.num_users], dtype=tf.float32),
        )


    def response(self, user_state, slate, item_state):
        # slate_indices: [U, K]
        slate_indices = slate.get('slate')
        features = item_state.get('features')                     # [N, T]
        gathered = tf.gather(features, slate_indices)             # [U, K, T]

        interest = user_state.get('interest')                     # [U, T]
        # Affinity v(s,i) = dot(user, item)
        affinities = tf.einsum('ut,ukt->uk', interest, gathered)  # [U, K]

        # Click model P(i|s,A) ~ softmax(beta * v)
        logits = self.beta * affinities                           # [U, K]
        choice = tf.cast(tf.random.categorical(logits, 1)[:, 0], tf.int32)  # [U]

        # Reward from chosen item
        reward_raw = tf.gather(affinities, choice[:, None], batch_dims=1)    # [U,1]
        reward_raw = tf.squeeze(reward_raw, axis=1)                          # [U]

        # Bound/scale reward for stability and metrics
        if self.reward_mode == "sigmoid":
            reward = tf.nn.sigmoid(reward_raw)                 # to (0,1)
        elif self.reward_mode == "clip01":
            # Min-max per-user over the slate to [0,1]
            a_min = tf.reduce_min(affinities, axis=1, keepdims=True)
            a_max = tf.reduce_max(affinities, axis=1, keepdims=True)
            denom = tf.maximum(a_max - a_min, 1e-6)
            norm = (affinities - a_min) / denom                # [U,K]
            reward = tf.gather(norm, choice[:, None], batch_dims=1)[:, 0]
        else:
            reward = tf.cast(reward_raw, tf.float32)           # "raw"

        return value.Value(choice=choice, reward=tf.cast(reward, tf.float32))


