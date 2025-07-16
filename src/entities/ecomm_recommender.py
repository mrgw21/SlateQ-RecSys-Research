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
        rec_features = tf.expand_dims(rec_state.get('rec_features'), axis=1)
        affinities = tf.matmul(user_state.get('interest'), rec_features)
        slate = tf.argsort(affinities, direction='DESCENDING')[:, :slate_size]
        return value.Value(slate=slate)

    def next_state(self, previous_state, response):
        return value.Value(rec_features=previous_state.get('rec_features'))