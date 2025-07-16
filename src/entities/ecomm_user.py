import gin
from recsim_ng.entities.state_models import static
from tensorflow_probability import distributions as tfd

@gin.configurable
class ECommUser(static.StaticStateModel):
    def __init__(self, num_topics=10, num_users=1):
        super().__init__()
        self.num_topics = num_topics
        self.num_users = num_users

    def specs(self):
        return {'interest': tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_users, self.num_topics))}

    def initial_state(self):
        return {'interest': tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_users, self.num_topics))}