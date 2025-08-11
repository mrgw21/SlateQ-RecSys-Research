from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from recsim_ng.core import value as value_lib


class ECommRuntime(runtime.TFRuntime):
    def __init__(self, network):
        super().__init__(network)
        self._current_state = self._network.initial_step()

        # Dynamically detect observation shapes for specs
        user_state = self._current_state.get("user_state", {})
        interest = user_state.get("interest")
        choice = user_state.get("choice")

        # Fallbacks for safety if keys not found
        self._num_users = int(interest.shape[0]) if interest is not None else 10
        self._interest_dim = int(interest.shape[1]) if interest is not None and len(interest.shape) > 1 else 10
        self._choice_shape = (self._num_users,) if choice is not None else (self._num_users,)

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
        reward = value.get("response").get("reward")
        choice = value.get("user_state").get("choice")

        return ts.transition(
            observation={
                "interest": interest,
                "choice": choice
            },
            reward=reward,
            discount=tf.ones_like(reward)
        )

    def observation_spec(self):
        return {
            "interest": tf.TensorSpec(shape=(self._num_users, self._interest_dim), dtype=tf.float32),
            "choice": tf.TensorSpec(shape=(self._num_users,), dtype=tf.int32),
        }

    def action_spec(self):
        return self._network.get_action_spec()

    def time_step_spec(self):
        return ts.TimeStep(
            step_type=tf.TensorSpec(shape=(self._num_users,), dtype=tf.int32),
            reward=tf.TensorSpec(shape=(self._num_users,), dtype=tf.float32),
            discount=tf.TensorSpec(shape=(self._num_users,), dtype=tf.float32),
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
