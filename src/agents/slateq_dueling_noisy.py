import tensorflow as tf
import tensorflow_probability as tfp
import copy

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib
from tf_agents.policies import tf_policy
from tf_agents.utils import common


import tensorflow as tf
from tf_agents.networks import network

class NoisyDense(tf.keras.layers.Layer):
    """Factorized NoisyNet layer (Fortunato et al., 2018)."""
    def __init__(self, units, activation=None, sigma_init=0.5, **kw):
        super().__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.sigma_init = float(sigma_init)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        mu_init = tf.keras.initializers.VarianceScaling(scale=2.0)
        self.w_mu = self.add_weight("w_mu", shape=[in_dim, self.units], initializer=mu_init)
        self.b_mu = self.add_weight("b_mu", shape=[self.units], initializer="zeros")

        self.w_sigma = self.add_weight("w_sigma", shape=[in_dim, self.units],
            initializer=tf.keras.initializers.Constant(self.sigma_init / tf.sqrt(tf.cast(in_dim, tf.float32))))
        self.b_sigma = self.add_weight("b_sigma", shape=[self.units],
            initializer=tf.keras.initializers.Constant(self.sigma_init / tf.sqrt(tf.cast(self.units, tf.float32))))

    @staticmethod
    def _f(x):
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def call(self, x, training=False):
        if training:
            eps_in  = tf.random.normal([x.shape[-1]])
            eps_out = tf.random.normal([self.units])
            f_in, f_out = self._f(eps_in), self._f(eps_out)
            w_eps = tf.tensordot(f_in, f_out, axes=0)
            w = self.w_mu + self.w_sigma * w_eps
            b = self.b_mu + self.b_sigma * f_out
            y = tf.linalg.matmul(x, w) + b
        else:
            y = tf.linalg.matmul(x, self.w_mu) + self.b_mu
        return self.activation(y) if self.activation is not None else y

# -------- SlateQNetwork with dueling + noisy --------
class SlateQNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 num_items,
                 fc_layer_params=(256, 128, 64),
                 dueling=True,
                 use_noisy=False,
                 l2=1e-5,
                 name='SlateQNetwork'):
        super().__init__(input_tensor_spec={'interest': input_tensor_spec}, state_spec=(), name=name)
        self._dueling = dueling
        self._num_items = int(num_items)
        self._use_noisy = use_noisy
        reg = tf.keras.regularizers.l2(l2) if l2 else None

        self._blocks = []
        for units in fc_layer_params:
            self._blocks.append(tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation=None,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
                    kernel_regularizer=reg),
                tf.keras.layers.LayerNormalization(epsilon=1e-5),
                tf.keras.layers.Activation('gelu')
            ]))

        # Heads
        Dense = NoisyDense if use_noisy else tf.keras.layers.Dense
        if dueling:
            self._adv = Dense(self._num_items, activation=None, kernel_regularizer=reg)
            self._val = Dense(1, activation=None, kernel_regularizer=reg)
        else:
            self._q = Dense(self._num_items, activation=None, kernel_regularizer=reg)

    def call(self, inputs, step_type=(), network_state=(), training=False):
        x = tf.convert_to_tensor(inputs['interest'])
        for blk in self._blocks:
            x = blk(x, training=training)

        if self._dueling:
            adv = self._adv(x, training=training)
            val = self._val(x, training=training)
            adv_centered = adv - tf.reduce_mean(adv, axis=-1, keepdims=True)
            q_values = val + adv_centered
        else:
            q_values = self._q(x, training=training)

        return q_values, network_state


class SlateQPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, q_network, slate_size):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._slate_size = slate_size

    def _distribution(self, time_step, policy_state):
        q_values, _ = self._q_network(time_step.observation, time_step.step_type)
        top_k = tf.math.top_k(q_values, k=self._slate_size)
        return tfp.distributions.Deterministic(loc=top_k.indices)

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


class SlateQExplorationPolicy(tf_policy.TFPolicy):
    def __init__(self,
                 base_policy,
                 num_items,
                 slate_size,
                 epsilon=0.3,
                 steps_to_min=60_000,
                 min_epsilon=0.10):
        super().__init__(base_policy.time_step_spec, base_policy.action_spec)
        self._base_policy = base_policy
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)

        self._epsilon = tf.Variable(float(epsilon), trainable=False, dtype=tf.float32)
        decay = (float(min_epsilon) / float(epsilon)) ** (1.0 / float(steps_to_min))
        self._epsilon_decay = tf.constant(decay, dtype=tf.float32)
        self._min_epsilon = tf.constant(float(min_epsilon), dtype=tf.float32)

    def _action(self, time_step, policy_state=(), seed=None):
        batch_size = tf.shape(time_step.observation['interest'])[0]

        rand_scores = tf.random.uniform([batch_size, self._num_items], dtype=tf.float32, seed=seed)
        random_slate = tf.math.top_k(rand_scores, k=self._slate_size).indices

        greedy_slate = self._base_policy.action(time_step).action

        explore_mask = tf.less(tf.random.uniform([batch_size], dtype=tf.float32, seed=seed),
                               self._epsilon)
        explore_mask = tf.expand_dims(explore_mask, 1)

        action = tf.where(explore_mask, random_slate, greedy_slate)
        return policy_step.PolicyStep(action=action, state=policy_state)

    def decay_epsilon(self, steps=1):
        if steps <= 0:
            return
        new_eps = self._epsilon * tf.pow(self._epsilon_decay, steps)
        self._epsilon.assign(tf.maximum(self._min_epsilon, new_eps))

    @property
    def epsilon(self):
        return float(self._epsilon.numpy())


class SlateQAgent(tf_agent.TFAgent):
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100,
                 learning_rate=1e-5,
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.05,
                 target_update_period=100,
                 gamma=0.95,
                 reward_scale=5.0):

        self._train_step_counter = tf.Variable(0)
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)
        self._target_update_period = int(target_update_period)

        input_tensor_spec = time_step_spec.observation['interest']

        # Main and target Q-networks
        self._q_network = SlateQNetwork(input_tensor_spec, num_items)
        self._target_q_network = SlateQNetwork(input_tensor_spec, num_items)

        # Build both nets once (dummy forward pass) BEFORE set_weights
        dummy_obs = {"interest": tf.zeros([1] + list(input_tensor_spec.shape), dtype=tf.float32)}
        dummy_step = tf.zeros([1], dtype=tf.int32)
        self._q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network(dummy_obs, dummy_step, training=False)

        # Optimizer
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Greedy base policy
        base_policy = SlateQPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=self._q_network,
            slate_size=slate_size
        )

        # Epsilon-greedy exploration policy
        self._policy = base_policy
        self._collect_policy = SlateQExplorationPolicy(
            base_policy=base_policy,
            num_items=num_items,
            slate_size=slate_size,
            epsilon=epsilon,
            steps_to_min=60_000,
            min_epsilon=min_epsilon
        )

        # TF-Agents base agent setup
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=self._policy,
            collect_policy=self._collect_policy,
            train_sequence_length=2,
            train_step_counter=self._train_step_counter
        )

        self._gamma = tf.constant(gamma, dtype=tf.float32)
        self._reward_scale = tf.constant(reward_scale, dtype=tf.float32)
        self._huber = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)

        # Sync target network weights with main network (now that they're built)
        self._target_q_network.set_weights(self._q_network.get_weights())

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
        return tf.group(
            *self._q_network.variables,
            *self._target_q_network.variables
        )

    def _train(self, experience, weights=None):
        # t0 / t1 slices
        obs_tm1 = tf.nest.map_structure(lambda x: x[:, 0], experience.observation)
        obs_tp1 = tf.nest.map_structure(lambda x: x[:, 1], experience.observation)
        actions   = experience.action[:, 1]
        rewards   = experience.reward[:, 1]
        discounts = tf.cast(experience.discount[:, 1], tf.float32)

        with tf.GradientTape() as tape:
            q_tm1, _ = self._q_network(obs_tm1, experience.step_type[:, 0])

            q_tp1_online, _ = self._q_network(obs_tp1, experience.step_type[:, 1])
            q_tp1_target, _  = self._target_q_network(obs_tp1, experience.step_type[:, 1])

            q_sa_items = tf.gather(q_tm1, actions, axis=1, batch_dims=1)
            q_sa = tf.reduce_mean(q_sa_items, axis=-1)

            # Double-Q target: argmax_k on ONLINE, evaluate with TARGET
            next_topk_idx = tf.math.top_k(q_tp1_online, k=self._slate_size).indices
            q_tp1_eval_items = tf.gather(q_tp1_target, next_topk_idx, axis=1, batch_dims=1)
            q_next_mean = tf.reduce_mean(q_tp1_eval_items, axis=-1)

            # Reward scaling
            r = tf.clip_by_value(rewards, 0.0, self._reward_scale) / self._reward_scale 

            # Target (masking by discount for terminals)
            target = tf.stop_gradient(r + self._gamma * discounts * q_next_mean)

            # Huber TD loss
            per_example_loss = self._huber(target, q_sa)
            if weights is not None:
                per_example_loss *= tf.cast(weights, tf.float32)

            reg_loss = tf.add_n(self._q_network.losses) if self._q_network.losses else 0.0
            loss = tf.reduce_mean(per_example_loss) + reg_loss

        # Optimizer step
        grads = tape.gradient(loss, self._q_network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self._optimizer.apply_gradients(zip(grads, self._q_network.trainable_variables))
        self._train_step_counter.assign_add(1)

        # Soft target update (Polyak)
        tau = 0.005
        for v, tv in zip(self._q_network.variables, self._target_q_network.variables):
            tv.assign(tau * v + (1.0 - tau) * tv)

        # Logging
        td_err = target - q_sa
        tf.print(
            "ε=", getattr(self._collect_policy, "_epsilon", -1.0),
            "Q[min,max]=", tf.reduce_min(q_tm1), tf.reduce_max(q_tm1),
            "TD|mean|=", tf.reduce_mean(tf.abs(td_err)),
            "Loss=", loss
        )

        return tf_agent.LossInfo(loss=loss, extra={"td_error": td_err})
