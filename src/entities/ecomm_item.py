import gin
from recsim_ng.entities.state_models import static
from tensorflow_probability import distributions as tfd
from recsim_ng.core import value
import tensorflow as tf

@gin.configurable
class ECommItem(static.StaticStateModel):
    def __init__(self, num_topics=10):
        super().__init__()
        self.num_topics = num_topics

    def specs(self):
        return value.ValueSpec(features=value.FieldSpec())

    def initial_state(self):
        return value.Value(features=tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_topics,)))

    def next_state(self, previous_state, response):
        return value.Value(features=previous_state['features'])