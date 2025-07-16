import gin
from recsim_ng.entities.state_models import static
from tensorflow_probability import distributions as tfd
from recsim_ng.core import value
import tensorflow as tf

@gin.configurable
class ECommItem(static.StaticStateModel):
    def __init__(self, num_topics=10, num_items=100):
        super().__init__()
        self.num_topics = num_topics
        self.num_items = num_items

    def specs(self):
        return {'features': value.ValueSpec(shape=(self.num_items, self.num_topics), dtype=tf.float32)}

    def initial_state(self):
        state_dict = {'features': tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_items, self.num_topics))}
        return value.Value(**state_dict)

    def next_state(self, previous_state, response):
        return previous_state