from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
import numpy as np

class ECommRuntime(runtime.TFRuntime):
    def __init__(self, network):
        super().__init__(network=network)

    def execute_with_rl(self, policy, num_steps=5000):
        driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(self, policy)
        driver.run(num_steps=num_steps)
        return self.current_value()

    def trajectory(self, num_steps=5000):
        def body(prev_state, step):
            return self._packed_step(prev_state)

        starting_value = self._network.initial_step()
        steps = tf.range(num_steps)

        # Define shape invariants based on the network's initial structure
        initial_state = self._network.initial_step()
        packed_initial = self._pack(initial_state)
        num_elements = len(packed_initial)
        shape_invariants = tuple(
            tf.TensorShape([]) if i == 0 else  # step counter
            tf.TensorShape([10, 10]) if i == 1 else  # user_state.interest
            tf.TensorShape([10]) if i in [2, 3] else  # item_state.features, rec_state.rec_features
            tf.TensorShape([5]) if i == 4 else  # slate
            tf.TensorShape([]) if i == 5 and num_elements > 5 else tf.TensorShape(None)  # response.choice (optional)
            for i in range(max(6, num_elements))
        ) + (tf.TensorShape([1]) if num_elements > 6 else tf.TensorShape(None),)  # response.reward (optional)
        print(f"Number of elements in packed initial state: {num_elements}, shape_invariants: {shape_invariants}")

        # Use tf.while_loop with shape_invariants
        def cond(step, _):
            return tf.less(step, num_steps)

        def body_with_step(step, state):
            next_state = body(state, step)
            # Print only at start, every 1000 steps, and end, adjusting for actual indices
            if step == 0 or step % 1000 == 0 or step == num_steps - 1:
                interest_idx = 1  # Assuming interest is at index 1
                response_idx = 5 if num_elements > 5 else None  # Start of response fields
                if interest_idx < len(next_state):
                    # Extract interest, handling potential dictionary misplacement
                    interest_val = next_state[interest_idx].get('interest', next_state[interest_idx]) if isinstance(next_state[interest_idx], dict) else next_state[interest_idx]
                    # Try to extract reward from response if available
                    reward_val = "N/A"
                    if response_idx is not None and response_idx < len(next_state) and isinstance(next_state[response_idx], (tf.Tensor, np.ndarray)):
                        reward_val = next_state[response_idx + 1] if response_idx + 1 < len(next_state) else "N/A"  # Reward at index 6
                    print(f"Step {step}: interest: {interest_val}, reward: {reward_val}")
                else:
                    print(f"Step {step}: interest: N/A, reward: N/A")
            return [step + 1, next_state]

        _, trajectory = tf.while_loop(
            cond,
            body_with_step,
            [0, self._pack(starting_value)],
            shape_invariants=shape_invariants
        )
        print(f"Completed {num_steps} steps")
        return self._unpack(trajectory)