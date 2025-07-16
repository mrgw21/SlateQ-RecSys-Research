import gin
from recsim_ng.entities.state_models import static
from tensorflow_probability import distributions as tfd

@gin.configurable
class ECommRecommender(static.StaticStateModel):
    def __init__(self, num_topics=10):
        super().__init__()
        self.num_topics = num_topics

    def specs(self):
        return {'rec_features': tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_topics,))}

    def initial_state(self):
        return {'rec_features': tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_topics,))}