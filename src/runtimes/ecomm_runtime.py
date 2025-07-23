from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from recsim_ng.core import value as value_lib


class ECommRuntime(runtime.TFRuntime):
    def __init__(self, network):
        super().__init__(network)
        self._current_state = self._network.initial_step()

    def reset(self):
        self._current_state = self._network.initial_step()
        return self._to_time_step(self._current_state)
    
    def step(self, action):
        input_dict = {
            "action": value_lib.Value(act=tf.convert_to_tensor(action)),
            "response": self._current_state["response"],
            "user_state": self._current_state["user_state"],
            "rec_state": self._current_state["rec_state"],
            "item_state": self._current_state["item_state"],
        }
        self._current_state = self._network.step(input_dict)

        # Convert to a proper TimeStep object
        return self._to_transition(self._current_state)

    def _to_time_step(self, value):
        interest = value.get("user_state").get("interest")
        num_users = tf.shape(interest)[0]

        return ts.TimeStep(
            step_type=tf.fill([num_users], ts.StepType.FIRST),
            reward=tf.zeros([num_users], dtype=tf.float32),
            discount=tf.ones([num_users], dtype=tf.float32),
            observation={"interest": interest}
        )

    def _to_transition(self, value):
        interest = value.get("user_state").get("interest")
        reward = value.get("response").get("resp")
        return ts.transition(
            observation={"interest": interest},
            reward=reward,
            discount=tf.ones_like(reward)
        )

    def observation_spec(self):
        return {
            "interest": tf.TensorSpec(shape=(10, 10), dtype=tf.float32)
        }

    def action_spec(self):
        return self._network.get_action_spec()

    def time_step_spec(self):
        return ts.TimeStep(
            step_type=tf.TensorSpec(shape=(10,), dtype=tf.int32),
            reward=tf.TensorSpec(shape=(10,), dtype=tf.float32),
            discount=tf.TensorSpec(shape=(10,), dtype=tf.float32),
            observation=self.observation_spec()
        )

    def trajectory(self, num_steps=5000):
        self._current_state = self._network.initial_step()
        packed_state = self._pack(self._current_state)

        def cond(step, *_):
            return tf.less(step, num_steps)

        def body(step, state):
            next_state = self._packed_step(state)
            return step + 1, next_state

        shape_invariants = [tf.TensorShape([])] + [
            tf.TensorShape(p.shape) if isinstance(p, tf.Tensor) else tf.TensorShape([])
            for p in packed_state
        ]

        _, final_state = tf.while_loop(
            cond, body, [0, packed_state], shape_invariants=shape_invariants
        )

        return self._unpack(final_state)
