import gin
import tensorflow as tf
from recsim_ng.entities.state_models.static import StaticStateModel
from recsim_ng.core import value


class TensorFieldSpec(value.FieldSpec):
    def __init__(self, shape, dtype):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def invariant(self):
        return tf.TensorSpec(shape=self.shape, dtype=self.dtype)


@gin.configurable
class ECommRecommender(StaticStateModel):
    def __init__(self, num_topics=10, num_users=10, slate_size=5):
        super().__init__()
        self.num_topics = num_topics
        self.num_users = num_users
        self.slate_size = slate_size

    def specs(self):
        return value.ValueSpec(
            rec_features=TensorFieldSpec(shape=(self.num_users, self.num_topics), dtype=tf.float32),
            slate=TensorFieldSpec(shape=(self.num_users, self.slate_size), dtype=tf.int32)
        )

    def initial_state(self):
        rec_features = tf.random.uniform(
            shape=(self.num_users, self.num_topics),
            minval=-1.0,
            maxval=1.0,
            dtype=tf.float32
        )
        slate = tf.zeros((self.num_users, self.slate_size), dtype=tf.int32)
        
        return value.Value(rec_features=rec_features, slate=slate)

    def next_state(self, previous_state, action):
        return previous_state

    def action(self, state, previous_action):
        slate = state.get('agent_slate')
        if slate is None:
            raise ValueError("Missing 'agent_slate' in state.")
        if slate.shape != (self.num_users, self.slate_size):
            raise ValueError(f"Expected agent_slate shape {(self.num_users, self.slate_size)}, got {slate.shape}")
        return value.Value(slate=slate)

    def action_spec(self):
        return tf.TensorSpec(shape=(self.num_users, self.slate_size), dtype=tf.int32, name="slate")
