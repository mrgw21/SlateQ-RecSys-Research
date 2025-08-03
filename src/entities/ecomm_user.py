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
    def __init__(self, num_topics=10, num_users=1):
        super().__init__()
        self.num_topics = num_topics
        self.num_users = num_users

    def specs(self):
        return ValueSpec(
            interest=TensorFieldSpec(shape=(self.num_users, self.num_topics), dtype=tf.float32),
            choice=TensorFieldSpec(shape=(self.num_users,), dtype=tf.int32),
            reward=TensorFieldSpec(shape=(self.num_users,), dtype=tf.float32)
        )

    def initial_state(self):
        interest = tfd.Normal(loc=0., scale=1.).sample(
            sample_shape=(self.num_users, self.num_topics)
        )

        choice = tf.zeros([self.num_users], dtype=tf.int32)
        reward = tf.zeros([self.num_users], dtype=tf.float32)

        return value.Value(interest=interest, choice=choice, reward=reward)

    def response(self, user_state, slate, item_state):
        slate_indices = slate.get('slate')
        features = item_state.get('features')
        gathered_features = tf.gather(features, slate_indices)

        interest = user_state.get('interest')
        interest_expanded = tf.expand_dims(interest, axis=1)

        affinities = tf.reduce_sum(interest_expanded * gathered_features, axis=-1)
        affinities = affinities / (tf.norm(affinities, axis=1, keepdims=True) + 1e-8)

        choice = tf.map_fn(
            lambda x: tfd.Categorical(logits=x).sample(),
            affinities,
            fn_output_signature=tf.int32
        )
        reward = tf.gather(affinities, choice[:, tf.newaxis], batch_dims=1)
        reward = tf.cast(reward, tf.float32)

        return value.Value(choice=choice, reward=reward)

    def response(self, user_state, slate, item_state):
        slate_indices = slate.get('slate')  # shape: [num_users, slate_size]
        features = item_state.get('features')

        # Gather features for the selected items in the slate
        gathered_features = tf.gather(features, slate_indices)  # [num_users, slate_size, num_topics]

        interest = user_state.get('interest')  # [num_users, num_topics]
        interest_expanded = tf.expand_dims(interest, axis=1)  # [num_users, 1, num_topics]

        affinities = tf.reduce_sum(interest_expanded * gathered_features, axis=-1)  # [num_users, slate_size]

        # Sample one item from slate
        choice = tf.map_fn(
            lambda x: tf.cast(tf.random.categorical(tf.expand_dims(x, 0), 1)[0, 0], tf.int32),
            affinities,
            fn_output_signature=tf.int32
        )

        # Reward = affinity of chosen item
        reward = tf.gather(affinities, choice[:, tf.newaxis], batch_dims=1)
        reward = tf.squeeze(reward, axis=1)

        return value.Value(choice=choice, reward=reward)


