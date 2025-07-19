import gin
from recsim_ng.entities.state_models.static import StaticStateModel
import tensorflow as tf
from recsim_ng.core import value

@gin.configurable
class ECommRecommender(StaticStateModel):
    def __init__(self, num_topics=10, num_users=10, slate_size=5):
        super().__init__()
        self.num_topics = num_topics
        self.num_users = num_users
        self.slate_size = slate_size

    def specs(self):
        return value.ValueSpec(
            rec_features=value.FieldSpec(),
            slate=value.FieldSpec()
        )

    def initial_state(self):
        rec_features = tf.random.uniform((self.num_users, self.num_topics), minval=-1.0, maxval=1.0)
        return value.Value(rec_features=rec_features)

    def next_state(self, previous_state, action):
        return previous_state  # Static for now

    def action(self, state, previous_action):
        # Expect slate from agent, raise error if missing
        slate = state.get('agent_slate')
        if slate is None or slate.shape != (self.num_users, self.slate_size):
            raise ValueError(f"Expected agent_slate with shape ({self.num_users}, {self.slate_size}), got {slate}")
        return value.Value(slate=slate)