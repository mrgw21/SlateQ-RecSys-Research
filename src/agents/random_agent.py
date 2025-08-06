import tensorflow as tf

from tf_agents.trajectories import policy_step
from tf_agents.policies import tf_policy

class RandomSlatePolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, num_items, slate_size):
        super().__init__(time_step_spec, action_spec)
        self._num_items = num_items
        self._slate_size = slate_size

    def _action(self, time_step, policy_state=(), seed=None):
        batch_size = tf.shape(time_step.observation["interest"])[0]
        action = tf.random.uniform(
            shape=(batch_size, self._slate_size),
            minval=0,
            maxval=self._num_items,
            dtype=tf.int32
        )
        return policy_step.PolicyStep(action=action, state=policy_state)
