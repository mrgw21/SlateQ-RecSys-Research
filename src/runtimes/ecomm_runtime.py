from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
import numpy as np

class ECommRuntime(runtime.TFRuntime):
    def __init__(self, network):
        super().__init__(network=network)

    def execute_with_rl(self, policy, num_steps=5000, num_episodes=10):
        for episode in range(num_episodes):
            time_step = self.reset()
            episode_reward = 0.0
            for step in range(num_steps):
                action_step = policy.action(time_step)
                next_time_step = self.step(action_step.action)
                experience = tf_agents.trajectories.trajectory.from_transition(time_step, action_step, next_time_step)
                # Train the agent
                if hasattr(policy._agent, '_train'):
                    loss_info = policy._agent.train(experience)
                    print(f"Episode {episode}, Step {step}, Loss: {loss_info.loss.numpy()}")
                episode_reward += next_time_step.reward.numpy().sum()
                time_step = next_time_step
            print(f"Episode {episode} completed, Total Reward: {episode_reward}")
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
            tf.TensorShape([10, 10]) if i == 5 and num_elements > 5 else tf.TensorShape(None)  # response.choice (per-topic)
            for i in range(max(6, num_elements))
        ) + (tf.TensorShape([10, 10, 1]) if num_elements > 6 else tf.TensorShape(None),)  # response.reward (per-topic)
        print(f"Number of elements in packed initial state: {num_elements}, shape_invariants: {shape_invariants}")

        # Use tf.while_loop with shape_invariants
        def cond(step, _):
            return tf.less(step, num_steps)

        def body_with_step(step, state):
            next_state = body(state, step)
            if step == 0 or step % 1000 == 0 or step == num_steps - 1:
                interest_idx = 1
                response_idx = 5 if num_elements > 5 else None
                if interest_idx < len(next_state):
                    interest_val = next_state[interest_idx] if isinstance(next_state[interest_idx], (tf.Tensor, np.ndarray)) else next_state[interest_idx].get('interest')
                    reward_val = "N/A"
                    if response_idx is not None and response_idx + 1 < len(next_state):
                        reward_val = next_state[response_idx + 1] if isinstance(next_state[response_idx + 1], (tf.Tensor, np.ndarray)) else "N/A"
                    print(f"Step {step}: interest: {interest_val}, reward: {reward_val}")
            return [step + 1, next_state]

        _, trajectory = tf.while_loop(
            cond,
            body_with_step,
            [0, self._pack(starting_value)],
            shape_invariants=shape_invariants
        )
        print(f"Completed {num_steps} steps")
        return self._unpack(trajectory)