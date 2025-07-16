from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

class ECommRuntime(runtime.TFRuntime):
    def __init__(self, network):
        super().__init__(network=network)

    def execute_with_rl(self, policy, num_steps=1000):
        driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(self, policy)
        driver.run(num_steps=num_steps)
        return self.current_value()

    def trajectory(self, num_steps=1000):
        def body(prev_state, step):
            return self._packed_step(prev_state)

        starting_value = self._network.initial_step()
        steps = tf.range(num_steps)

        # Define shape invariants as a tuple of TensorShape objects for step and each state component
        shape_invariants = (
            tf.TensorShape([]),  # step counter
            tf.TensorShape([10, 10]),  # user_state.interest
            tf.TensorShape([10]),      # item_state.features
            tf.TensorShape([10]),      # rec_state.rec_features
            tf.TensorShape([5]),       # slate
            tf.TensorShape([]),        # response.choice
            tf.TensorShape([1])        # response.reward (updated to [1] for (10, 1))
        )

        # Use tf.while_loop with shape_invariants
        def cond(step, _):
            return tf.less(step, num_steps)

        def body_with_step(step, state):
            next_state = body(state, step)
            # Convert step to numpy value safely, or avoid .numpy() if not needed
            step_value = step  # Use step directly as a tensor, or use step.numpy() in eager mode
            print(f"Body step {step_value}, next_state interest: {next_state[1]}")  # Debug state update
            return [step + 1, next_state]

        _, trajectory = tf.while_loop(
            cond,
            body_with_step,
            [0, self._pack(starting_value)],
            shape_invariants=shape_invariants
        )
        return self._unpack(trajectory)