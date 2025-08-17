import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.trajectories import policy_step, trajectory as trajectory_lib
from tf_agents.policies import tf_policy


# Vanilla per-item Q network
class VanillaSlateQNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 num_items: int,
                 hidden=(256, 128, 64),
                 l2=0.0,
                 name='VanillaSlateQNetwork'):
        super().__init__(input_tensor_spec={'interest': input_tensor_spec}, state_spec=(), name=name)
        self._num_items = int(num_items)
        reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None

        self._mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(h, activation='relu',
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
                                  kernel_regularizer=reg)
            for h in hidden
        ])
        self._head = tf.keras.layers.Dense(self._num_items, activation=None, kernel_regularizer=reg)

    def call(self, inputs, step_type=(), network_state=(), training=False):
        x = tf.convert_to_tensor(inputs['interest'])
        x = self._mlp(x, training=training)
        q_values = self._head(x, training=training)
        return q_values, network_state


# Greedy top-k policy (rank by v * Q)
class SlateQPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, q_network, slate_size: int, beta: float = 5.0):
        super().__init__(time_step_spec, action_spec)
        self._q_network = q_network
        self._slate_size = int(slate_size)
        self._beta = float(beta)

    def _distribution(self, time_step, policy_state):
        obs = time_step.observation

        q_values, _ = self._q_network(obs, time_step.step_type)

        interest   = tf.convert_to_tensor(obs["interest"])
        item_feats = tf.convert_to_tensor(obs["item_features"])

        item_feats = tf.cond(
            tf.equal(tf.rank(item_feats), 2),
            lambda: tf.tile(tf.expand_dims(item_feats, 0), [tf.shape(interest)[0], 1, 1]),
            lambda: item_feats
        )

        v_all = tf.einsum("bd,bnd->bn", interest, item_feats)

        score = v_all * q_values
        top_k = tf.math.top_k(score, k=self._slate_size)

        return tfp.distributions.Deterministic(loc=top_k.indices)

    def _action(self, time_step, policy_state, seed=None):
        action = self._distribution(time_step, policy_state).sample(seed=seed)
        return policy_step.PolicyStep(action=action, state=policy_state)


# Epsilon-greedy exploration
class SlateQExplorationPolicy(tf_policy.TFPolicy):
    def __init__(self,
                 base_policy: SlateQPolicy,
                 num_items: int,
                 slate_size: int,
                 epsilon: float = 0.2,
                 steps_to_min: int = 20_000,
                 min_epsilon: float = 0.05):
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

        explore_mask = tf.less(tf.random.uniform([batch_size], dtype=tf.float32, seed=seed), self._epsilon)
        explore_mask = tf.expand_dims(explore_mask, 1)

        action = tf.where(explore_mask, random_slate, greedy_slate)
        return policy_step.PolicyStep(action=action, state=policy_state)

    def decay_epsilon(self, steps=1):
        if steps <= 0:
            return
        new_eps = self._epsilon * tf.pow(self._epsilon_decay, steps)
        self._epsilon.assign(tf.maximum(self._min_epsilon, new_eps))

    @property
    def epsilon(self) -> float:
        return float(self._epsilon.numpy())


# -----------
# Vanilla Agent
# -----------
class SlateQAgent(tf_agent.TFAgent):
    """
    Vanilla SlateQ:
      - Q-network produces Q(s, i) for all items.
      - Only updates the clicked item.
      - Expected next-slate target using softmax over target Q(s', j) on the greedy next slate.
      - Hard target update every target_update_period steps.
    """
    def __init__(self, time_step_spec, action_spec,
                 num_users=10, num_topics=10, slate_size=5, num_items=100,
                 learning_rate=1e-3,
                 epsilon=0.2, min_epsilon=0.05, epsilon_decay_steps=20_000,
                 target_update_period=1000,
                 gamma=0.95,
                 beta=5.0,
                 huber_delta=1.0,
                 l2=0.0):

        self._train_step_counter = tf.Variable(0)
        self._num_items = int(num_items)
        self._slate_size = int(slate_size)
        self._target_update_period = int(target_update_period)
        self._gamma = tf.constant(gamma, dtype=tf.float32)
        self._beta = tf.constant(beta, dtype=tf.float32)
        self.is_learning = True

        input_tensor_spec = time_step_spec.observation['interest']

        # Main and target Q-networks
        self._q_network = VanillaSlateQNetwork(input_tensor_spec, num_items, l2=l2)
        self._target_q_network = VanillaSlateQNetwork(input_tensor_spec, num_items, l2=l2)

        # Build both nets once before copying weights
        dummy_obs = {"interest": tf.zeros([1] + list(input_tensor_spec.shape), dtype=tf.float32)}
        dummy_step = tf.zeros([1], dtype=tf.int32)
        self._q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network(dummy_obs, dummy_step, training=False)
        self._target_q_network.set_weights(self._q_network.get_weights())

        # Optimizer
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._huber = tf.keras.losses.Huber(delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)

        # Greedy base policy
        base_policy = SlateQPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=self._q_network,
            slate_size=slate_size,
            beta=float(self._beta.numpy()) if hasattr(self._beta, "numpy") else self._beta,
        )

        # Epsilon-greedy exploration policy
        self._policy = base_policy
        self._collect_policy = SlateQExplorationPolicy(
            base_policy=base_policy,
            num_items=num_items,
            slate_size=slate_size,
            epsilon=epsilon,
            steps_to_min=epsilon_decay_steps,
            min_epsilon=min_epsilon
        )

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=self._policy,
            collect_policy=self._collect_policy,
            train_sequence_length=2,
            train_step_counter=self._train_step_counter
        )

    # For replay buffer dtype
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
        return tf.no_op()

    def _train(self, experience, weights=None):
        def slice_time(x, t_idx: int):
            return x[:, t_idx]

        def flatten_keep_tail(x, tail_ndims: int):
            x = tf.convert_to_tensor(x)
            shape = tf.shape(x)
            rank = tf.rank(x)

            def flatten_all():
                return tf.reshape(x, [-1])

            def flatten_keep():
                if tail_ndims == 0:
                    return tf.reshape(x, [-1])
                lead = tf.reduce_prod(shape[:-tail_ndims])
                tail = shape[-tail_ndims:]
                return tf.reshape(x, tf.concat([[lead], tail], axis=0))

            return tf.cond(tf.less(rank, tail_ndims), flatten_all, flatten_keep)

        obs_t_interest    = slice_time(experience.observation['interest'], 0)
        obs_tp1_interest  = slice_time(experience.observation['interest'], 1)
        click_pos_t       = slice_time(experience.observation['choice'], 1)

        item_feats_tp1_any = slice_time(experience.observation['item_features'], 1)
        item_feats_rank = tf.rank(item_feats_tp1_any)
        item_feats_tp1 = tf.case([
            (tf.equal(item_feats_rank, 4), lambda: item_feats_tp1_any[0, 0]),
            (tf.equal(item_feats_rank, 3), lambda: item_feats_tp1_any[0]),
        ], default=lambda: item_feats_tp1_any)

        act = experience.action
        act_rank = tf.rank(act)
        action_t = tf.cond(
            tf.greater_equal(act_rank, 2),
            lambda: act[:, 0],
            lambda: act
        )

        reward_t      = slice_time(experience.reward, 1)
        discount_tp1  = tf.cast(slice_time(experience.discount, 1), tf.float32)

        obs_t   = {'interest': flatten_keep_tail(obs_t_interest,   tail_ndims=1)}
        obs_tp1 = {'interest': flatten_keep_tail(obs_tp1_interest, tail_ndims=1)}
        action_t     = flatten_keep_tail(action_t,     tail_ndims=1)
        click_pos_t  = tf.cast(flatten_keep_tail(click_pos_t,  tail_ndims=0), tf.int32)
        reward_t     = flatten_keep_tail(reward_t,     tail_ndims=0)
        discount_tp1 = flatten_keep_tail(discount_tp1, tail_ndims=0)

        batch_size = tf.shape(action_t)[0]

        idx = tf.stack([tf.range(batch_size), click_pos_t], axis=1)
        clicked_item_id = tf.gather_nd(action_t, idx)

        with tf.GradientTape() as tape:
            # Q(s_t, :)
            q_tm1_all, _ = self._q_network(obs_t, training=True)
            q_clicked = tf.gather(q_tm1_all, clicked_item_id, axis=1, batch_dims=1)

            q_tp1_online_all, _ = self._q_network(obs_tp1, training=False)

            v_all = tf.linalg.matmul(obs_tp1['interest'], item_feats_tp1, transpose_b=True)
            score_next = v_all * q_tp1_online_all
            next_topk_idx = tf.math.top_k(score_next, k=self._slate_size).indices


            q_tp1_target_all, _ = self._target_q_network(obs_tp1, training=False)
            q_next_on_slate = tf.gather(q_tp1_target_all, next_topk_idx, axis=1, batch_dims=1)

            v_all = tf.linalg.matmul(obs_tp1['interest'], item_feats_tp1, transpose_b=True)
            v_next_on_slate = tf.gather(v_all, next_topk_idx, axis=1, batch_dims=1)
            p_next = tf.nn.softmax(self._beta * v_next_on_slate, axis=-1)
            expect_next = tf.reduce_sum(p_next * q_next_on_slate, axis=-1)

            # TD target
            y = tf.stop_gradient(reward_t + self._gamma * discount_tp1 * expect_next)

            # Huber on clicked item only
            per_example = tf.keras.losses.huber(y, q_clicked, delta=1.0)
            loss = tf.reduce_mean(per_example)

        # Optimiser step
        grads = tape.gradient(loss, self._q_network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self._optimizer.apply_gradients(zip(grads, self._q_network.trainable_variables))
        self._train_step_counter.assign_add(1)

        # Hard target update
        step = int(self._train_step_counter.numpy())
        if step % self._target_update_period == 0:
            self._target_q_network.set_weights(self._q_network.get_weights())

        return tf_agent.LossInfo(loss=loss, extra={})
