import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from src.core.registry import register

class _RandomPolicy:
    def __init__(self, action_spec):
        self._action_spec = action_spec
    def action(self, time_step: ts.TimeStep):
        batch = tf.shape(time_step.step_type)[0]
        slate_size = self._action_spec.shape[0]
        maxv = int(self._action_spec.maximum)
        action = tf.random.uniform(
            shape=(batch, slate_size), minval=0, maxval=maxv + 1, dtype=tf.int32
        )
        return policy_step.PolicyStep(action=action)

    def decay_epsilon(self, steps=1):
        return

class RandomAgentShim:
    def __init__(self, time_step_spec, action_spec, **_kwargs):
        self.policy = _RandomPolicy(action_spec)
        self.collect_policy = self.policy
        self.is_learning = False
        self.collect_data_spec = None

@register("random")
def make_random(time_step_spec, action_spec, **kwargs):
    return RandomAgentShim(time_step_spec, action_spec, **kwargs)
