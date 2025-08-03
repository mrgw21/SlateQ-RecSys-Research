import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib
from tf_agents.policies import tf_policy
from tf_agents.utils import common


class SlateQNetwork(network.Network):
    def __init__(self, input_tensor_spec, num_items, fc_layer_params=(128, 64), name='SlateQNetwork'):
        super().__init__(input_tensor_spec={'interest': input_tensor_spec}, state_spec=(), name=name)
        self._num_items = num_items
        self._fc_layers = [tf.keras.layers.Dense(units, activation='relu') for units in fc_layer_params]
        self._output_layer = tf.keras.layers.Dense(num_items, activation=None)

    def call(self, inputs, step_type=(), network_state=(), training=False):
        x = inputs['interest']
        for layer in self._fc_layers:
            x = layer(x, training=training)
        q_values = self._output_layer(x)
        return q_values, network_state


class SlateQPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, q_network, slate_size):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._slate_size = slate_size

    def _distribution(self, time_step, policy_state):
        q_values, _ = self._q_network(time_step.observation, time_step.step_type)
        top_k_indices = tf.math.top_k(q_values, k=self._slate_size).indices  # [batch_size, slate_size]
        return tfp.distributions.Deterministic(loc=top_k_indices)

    def _action(self, time_step, policy_state, seed=None):
        dist = self._distribution(time_step, policy_state)
        action = dist.sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


class SlateQExplorationPolicy(tf_policy.TFPolicy):
    def __init__(self, base_policy, num_items, slate_size, epsilon=0.1):
        super().__init__(base_policy.time_step_spec, base_policy.action_spec)
        self._base_policy = base_policy
        self._num_items = num_items
        self._slate_size = slate_size
        self._epsilon = epsilon

    def _action(self, time_step, policy_state=(), seed=None):
        batch_size = tf.shape(time_step.observation['interest'])[0]

        def random_slate():
            return tf.random.uniform(
                shape=(batch_size, self._slate_size),
                minval=0,
                maxval=self._num_items,
                dtype=tf.int32
            )

        def greedy_slate():
            return self._base_policy.action(time_step).action

        should_explore = tf.less(tf.random.uniform([batch_size]), self._epsilon)
        action = tf.where(should_explore[:, tf.newaxis], random_slate(), greedy_slate())
        return policy_step.PolicyStep(action=action, state=policy_state)


class SlateQAgent(tf_agent.TFAgent):
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100, learning_rate=1e-3):
        self._train_step_counter = tf.Variable(0)

        self._num_items = num_items
        self._slate_size = slate_size

        input_tensor_spec = time_step_spec.observation['interest']

        self._q_network = SlateQNetwork(input_tensor_spec, num_items)

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        base_policy = SlateQPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=self._q_network,
            slate_size=slate_size
        )

        self._policy = base_policy
        self._collect_policy = SlateQExplorationPolicy(
            base_policy=base_policy,
            num_items=num_items,
            slate_size=slate_size,
            epsilon=0.1
        )

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=self._policy,
            collect_policy=self._collect_policy,
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
        obs_tm1 = tf.nest.map_structure(lambda x: x[:, 0], experience.observation)
        obs_tp1 = tf.nest.map_structure(lambda x: x[:, 1], experience.observation)
        rewards = experience.reward[:, 1]
        actions = experience.action[:, 1]

        with tf.GradientTape() as tape:
            # Q(s, a)
            q_tm1, _ = self._q_network(obs_tm1, experience.step_type[:, 0])
            a_one_hot = tf.one_hot(actions, depth=self._num_items)

            q_tm1_exp = tf.expand_dims(q_tm1, axis=2)
            q_tm1_exp = tf.broadcast_to(q_tm1_exp, tf.shape(a_one_hot))

            q_action_values = tf.reduce_sum(q_tm1_exp * tf.cast(a_one_hot, tf.float32), axis=-1)
            q_sa = tf.reduce_mean(q_action_values, axis=-1)

            # Q(s', a') for next state
            q_tp1, _ = self._q_network(obs_tp1, experience.step_type[:, 1])
            top_k = tf.math.top_k(q_tp1, k=self._slate_size).indices
            top_k_one_hot = tf.one_hot(top_k, depth=self._num_items)

            q_tp1_exp = tf.expand_dims(q_tp1, axis=2)
            q_tp1_exp = tf.broadcast_to(q_tp1_exp, tf.shape(top_k_one_hot))

            q_top_k = tf.reduce_sum(q_tp1_exp * tf.cast(top_k_one_hot, tf.float32), axis=-1)
            q_next_mean = tf.reduce_mean(q_top_k, axis=-1)

            target = rewards + 0.99 * q_next_mean
            loss = tf.reduce_mean(tf.square(target - q_sa))

        grads = tape.gradient(loss, self._q_network.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._q_network.trainable_variables))
        self._train_step_counter.assign_add(1)

        return tf_agent.LossInfo(loss=loss, extra={})

