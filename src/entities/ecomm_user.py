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
            interest=TensorFieldSpec(shape=(self.num_users, self.num_topics), dtype=tf.float32)
        )

    def initial_state(self):
        interest = tfd.Normal(loc=0., scale=1.).sample(
            sample_shape=(self.num_users, self.num_topics)
        )
        return value.Value(interest=interest)

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

    def next_state(self, previous_state, response):
        reward = tf.clip_by_value(response.get('resp'), -1.0, 1.0)
        reward = tf.reshape(reward, [self.num_users, 1])
        decay_factor = 0.99
        interest = previous_state.get('interest')
        updated_interest = decay_factor * interest + 0.1 * reward
        updated_interest = tf.clip_by_value(updated_interest, -10.0, 10.0)
        return value.Value(interest=updated_interest)
