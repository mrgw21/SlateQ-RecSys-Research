import gin
from recsim_ng.entities.state_models import static
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from recsim_ng.core import value

@gin.configurable
class ECommRecommender(static.StaticStateModel):
    def __init__(self, num_topics=10):
        super().__init__()
        self.num_topics = num_topics

    def specs(self):
        return value.ValueSpec(rec_features=value.FieldSpec())

    def initial_state(self):
        return value.Value(rec_features=tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_topics,)))

    def select_slate(self, rec_state, user_state, slate_size):
        # Ensure diverse slate by using user interest to rank items
        rec_features = rec_state.get('rec_features')  # Shape: (10,)
        user_interest = user_state.get('interest')    # Shape: (10, 10)
        # Compute affinities with proper broadcasting
        affinities = tf.tensordot(user_interest, rec_features, axes=1)  # Shape: (10,) per user
        # Sort and select top slate_size indices for each user
        top_indices = tf.argsort(affinities, direction='DESCENDING')[:slate_size]  # Shape: (slate_size,)
        # Create a (10, slate_size) slate by repeating for all users using tf.repeat
        top_indices_tiled = tf.repeat(tf.expand_dims(top_indices, axis=0), tf.shape(user_interest)[0], axis=0)  # Shape: (10, slate_size)
        return value.Value(slate=top_indices_tiled)

    def next_state(self, previous_state, response):
        return value.Value(rec_features=previous_state.get('rec_features'))