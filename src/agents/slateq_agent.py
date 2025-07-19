# agents/slateq_agent.py
from tf_agents.agents import tf_agent
from tf_agents.networks import q_network
from tf_agents.policies import q_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
import tensorflow as tf
import numpy as np

class SlateQPolicy(q_policy.QPolicy):
    def __init__(self, time_step_spec, action_spec, q_network):
        super().__init__(time_step_spec, action_spec, q_network=q_network)

    def _distribution(self, time_step):
        # Compute Q-values for all possible actions (simplified to slate_size per user)
        q_values = self._q_network(time_step.observation)
        batch_size = tf.shape(time_step.observation)[0]
        # Select top slate_size actions per user
        action_indices = tf.argsort(q_values, axis=-1, direction='DESCENDING')[:, :self._action_spec.shape[1]]
        # Reshape to match slate action spec (batch_size, slate_size)
        action = tf.reshape(action_indices, [batch_size, -1])
        return tfp.distributions.Deterministic(loc=action)

    def _action(self, time_step, policy_state, seed=None):
        distribution = self._distribution(time_step)
        action = tf.cast(distribution.sample(), dtype=self._action_spec.dtype)
        return tf_agent.PolicyStep(action, policy_state)

class SlateQAgent(tf_agent.TFAgent):
    def __init__(self, time_step_spec, action_spec, num_users=10, num_topics=10, slate_size=5, learning_rate=1e-3):
        super().__init__(time_step_spec, action_spec, train_sequence_length=2)

        # Define observation and action specs
        self.num_users = num_users
        self.num_topics = num_topics
        self.slate_size = slate_size

        # Q-network for slate actions
        fc_layer_params = (100, 50)
        self.q_network = q_network.QNetwork(
            input_tensor_spec=time_step_spec.observation,
            action_spec=action_spec,
            fc_layer_params=fc_layer_params,
            activation_fn=tf.nn.relu
        )

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Global step for training
        train_step_counter = tf.Variable(0)

        # Initialize the agent with the custom policy
        self._policy = SlateQPolicy(time_step_spec, action_spec, q_network=self.q_network)
        super().__init__(
            time_step_spec,
            action_spec,
            policy=self._policy,
            collect_policy=self._policy,  # Use the same policy for collection and action
            train_step_counter=train_step_counter,
            optimizer=optimizer,
            gradient_clipping=None,
            debug_summaries=False,
            summarize_grads_and_vars=False,
            train_sequence_length=2
        )

    def _train(self, experience, weights=None):
        with tf.GradientTape() as tape:
            q_values = self.q_network(experience.observation)
            # Simplified Q-loss (needs slate-specific adjustment)
            loss = tf.reduce_mean(tf.square(experience.reward - q_values))
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        self.train_step_counter.assign_add(1)
        return tf_agent.LossInfo(loss=loss, extra={})

    def _initialize(self):
        tf.nest.map_structure(lambda x: x, self.q_network.variables)