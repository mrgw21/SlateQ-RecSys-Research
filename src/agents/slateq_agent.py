import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.policies import tf_policy
from tf_agents.utils import common
from tf_agents.trajectories import trajectory as trajectory_lib


class SlateQNetwork(network.Network):
    def __init__(self, input_tensor_spec, num_items, fc_layer_params=(128, 64), name='SlateQNetwork'):
        super().__init__(input_tensor_spec={'interest': input_tensor_spec}, state_spec=(), name=name)
        self._num_items = num_items
        self._fc_layers = [tf.keras.layers.Dense(units, activation='relu') for units in fc_layer_params]
        self._output_layer = tf.keras.layers.Dense(num_items, activation=None)

    def call(self, inputs, step_type=(), network_state=()):
        x = inputs['interest']
        for layer in self._fc_layers:
            x = layer(x)
        q_values = self._output_layer(x)
        return q_values, network_state

class SlateQPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, q_network, slate_size, num_users):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._slate_size = slate_size
        self._num_users = num_users

    def _distribution(self, time_step, policy_state):
        q_values, _ = self._q_network(time_step.observation, time_step.step_type)
        top_k_indices = tf.math.top_k(q_values, k=self._slate_size).indices  # [batch_size, slate_size]
        return tfp.distributions.Deterministic(loc=top_k_indices)

    def _action(self, time_step, policy_state, seed=None):
        dist = self._distribution(time_step, policy_state)
        action = dist.sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


class SlateQAgent(tf_agent.TFAgent):
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100, learning_rate=1e-3):
        self._train_step_counter = tf.Variable(0)

        input_tensor_spec = time_step_spec.observation['interest']
        self._num_items = num_items
        self._slate_size = slate_size

        self._q_network = SlateQNetwork(
            input_tensor_spec=input_tensor_spec,
            num_items=num_items,
            fc_layer_params=(128, 64),
            name='SlateQNetwork'
        )

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self._policy = SlateQPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=self._q_network,
            slate_size=slate_size,
            num_users=num_users
        )

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=self._policy,
            collect_policy=self._policy,
            train_sequence_length=2,
            train_step_counter=self._train_step_counter
        )

    @property
    def collect_data_spec(self):
        return trajectory_lib.Trajectory(
            step_type=self._time_step_spec.step_type,
            observation=self._time_step_spec.observation,
            action=self._action_spec,
            policy_info=(),
            next_step_type=self._time_step_spec.step_type,
            reward=self._time_step_spec.reward,
            discount=self._time_step_spec.discount,
        )

    def _initialize(self):
        return tf.group(self._q_network.variables)

    def _train(self, experience, weights=None):
        # Slice timestep=1 (latest)
        observations = tf.nest.map_structure(lambda x: x[:, -1], experience.observation)
        rewards = experience.reward[:, -1]         # [batch, users]
        actions = experience.action[:, -1]         # [batch, users, slate]
        step_type = experience.step_type[:, -1]

        with tf.GradientTape() as tape:
            # Q-values: [batch, users, items]
            q_values, _ = self._q_network(observations, step_type)

            # Broadcast to [batch, users, slate, items]
            q_values_tiled = tf.tile(tf.expand_dims(q_values, axis=2), [1, 1, self._slate_size, 1])

            # One-hot encode actions: [batch, users, slate, items]
            actions_one_hot = tf.one_hot(actions, depth=self._num_items)

            # Select Q-values for taken actions: [batch, users, slate]
            selected_q_values = tf.reduce_sum(q_values_tiled * tf.cast(actions_one_hot, tf.float32), axis=-1)

            # Mean Q over slate: [batch, users]
            selected_q_values_mean = tf.reduce_mean(selected_q_values, axis=-1)

            # Loss: MSE between predicted Q and reward
            loss = tf.reduce_mean(tf.square(rewards - selected_q_values_mean))

        gradients = tape.gradient(loss, self._q_network.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._q_network.trainable_variables))
        self._train_step_counter.assign_add(1)

        return tf_agent.LossInfo(loss=loss, extra={})
