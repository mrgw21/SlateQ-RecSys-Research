from recsim_ng.entities.state_models import static
from tensorflow_probability import distributions as tfd

class ECommUser(static.StaticStateModel):
    def __init__(self, num_topics=10):
        self.num_topics = num_topics

    def specs(self):
        return {'interest': tfd.Normal(loc=0., scale=1.).sample(shape=(self.num_topics,))}

    def initial_state(self):
        return {'interest': tfd.Normal(loc=0., scale=1.).sample(shape=(self.num_topics,))}