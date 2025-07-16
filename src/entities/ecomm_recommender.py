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
        return value.ValueSpec(fields={'rec_features': value.ValueSpec(shape=(self.num_topics,), dtype=tf.float32)})

    def initial_state(self):
        state_dict = {'rec_features': tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_topics,))}
        return value.Value(**state_dict)

    def select_slate(self, rec_state, user_state, slate_size):
        affinities = tf.matmul(user_state['interest'], rec_state['rec_features'], transpose_b=True)
        slate = tf.argsort(affinities, direction='DESCENDING')[:, :slate_size]
        return slate

    def next_state(self, previous_state, response):
        return previous_state