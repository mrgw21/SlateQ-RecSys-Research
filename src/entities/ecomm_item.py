import gin
from recsim_ng.entities.state_models import static
from tensorflow_probability import distributions as tfd
from recsim_ng.core import value
import tensorflow as tf

class TensorFieldSpec(value.FieldSpec):
    def __init__(self, shape, dtype=tf.float32):
        super().__init__()
        self._shape = shape
        self._dtype = dtype

    def invariant(self):
        return tf.TensorSpec(shape=self._shape, dtype=self._dtype)

@gin.configurable
class ECommItem(static.StaticStateModel):
    def __init__(self, num_topics=10):
        super().__init__()
        self.num_topics = num_topics

    def specs(self):
        return value.ValueSpec(
            features=TensorFieldSpec(shape=(self.num_topics,), dtype=tf.float32)
        )

    def initial_state(self):
        features = tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_topics,))
        return value.Value(features=features)

    def next_state(self, previous_state, response):
        # Static state, no update
        return value.Value(features=previous_state.get('features'))
