import gin
from recsim_ng.entities.state_models import static
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from recsim_ng.core import value

@gin.configurable
class ECommUser(static.StaticStateModel):
    def __init__(self, num_topics=10, num_users=1):
        super().__init__()
        self.num_topics = num_topics
        self.num_users = num_users

    def specs(self):
        return value.ValueSpec(fields={'interest': value.ValueSpec(shape=(self.num_users, self.num_topics), dtype=tf.float32)})

    def initial_state(self):
        state_dict = {'interest': tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_users, self.num_topics))}
        return value.Value(**state_dict)

    def response(self, user_state, slate, item_state):
        affinities = tf.matmul(user_state['interest'], item_state['features'][slate.value], transpose_b=True)
        choice = tfd.Categorical(logits=affinities).sample()
        return {'choice': choice, 'reward': affinities[choice]}

    def next_state(self, previous_state, response):
        return {'interest': previous_state['interest'] + 0.1 * response.value['reward']}