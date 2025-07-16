import gin
from recsim_ng.entities.state_models import static
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from recsim_ng.core import value

@gin.configurable
class ECommUser(static.StaticStateModel):
    def __init__(self, num_topics=10, num_users=1):
        super().__init__()
        self.num_topics = num_topics
        self.num_users = num_users

    def specs(self):
        # Define specs for initial state only
        return value.ValueSpec(interest=value.FieldSpec())

    def initial_state(self):
        interest = tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_users, self.num_topics))
        print(f"Initial interest raw shape: {interest.shape}, values: {interest}")  # Keep initial state print
        return value.Value(interest=interest)  # Only interest in initial state

    def response(self, user_state, slate, item_state):
        slate_indices = slate.get('slate')  # Shape: (10, slate_size) = (10, 5)
        features = item_state.get('features')  # Shape: (num_topics,) = (10,)
        # Gather features using slate indices
        gathered_features = tf.gather(features, slate_indices)  # Should be (10, 5)
        gathered_features = tf.expand_dims(gathered_features, axis=-1)  # Shape: (10, 5, 1)
        gathered_features = tf.squeeze(gathered_features, axis=None)  # Remove extra singleton dims
        # Force reshape to (10, 5, 1) if the shape is incorrect
        if gathered_features.shape[-1] != 1:
            gathered_features = tf.reshape(gathered_features, [self.num_users, -1, 1])  # Reshape to (10, 5, 1)
        # Handle potential extra dimensions from runtime for interest
        interest = user_state.get('interest')  # Shape: (10, 10) or higher rank
        interest = tf.squeeze(interest)  # Remove all singleton dimensions
        # Handle 4D case (e.g., batch of users or time steps)
        if len(interest.shape) == 4 and interest.shape[0] == self.num_users and interest.shape[2] == self.num_topics:
            interest = interest[:, 0, :, 0]  # Take the first batch/user slice and last dimension
        elif len(interest.shape) == 3 and interest.shape[0] == self.num_users:
            interest = interest[:, :, 0]  # Take the first slice if it's a batch of interests
        elif len(interest.shape) == 3 and interest.shape[1] == self.num_topics:
            interest = interest[0, :, :]  # Take the first user if stacked by users
        elif len(interest.shape) != 2:
            raise ValueError(f"Interest shape {interest.shape} is not reducible to rank 2 after squeeze")
        print(f"Processed interest shape: {interest.shape}, values: {interest}")  # Debug processed shape
        interest_expanded = tf.expand_dims(interest, axis=-1)  # Shape: (10, 10, 1), ensure rank 3
        # Compute affinities per topic-topic pair
        affinities = tf.einsum('bij,bkj->bik', interest_expanded, gathered_features)  # Shape: (10, 10, 5)
        # Normalize affinities per user and topic
        max_abs_affinities = tf.reduce_max(tf.abs(affinities), axis=2, keepdims=True) + 1e-10  # Max per user and topic
        affinities = affinities / max_abs_affinities  # Normalize per user and topic
        # Sample choice per topic
        choice = tf.map_fn(lambda x: tfd.Categorical(logits=x).sample(), affinities, fn_output_signature=tf.int32)  # Shape: (10, 10)
        # Derive reward per topic
        reward = tf.gather(affinities, choice, batch_dims=1, axis=2)  # Shape: (10, 10)
        reward = tf.expand_dims(reward, axis=-1)  # Shape: (10, 10, 1)
        return value.Value(choice=choice, reward=reward)  # Return per-topic choices and rewards

    def next_state(self, previous_state, response):
        reward = tf.clip_by_value(response.get('reward'), -1.0, 1.0)  # Clip reward to prevent extreme values
        # Ensure reward shape matches interest for broadcasting
        reward = tf.squeeze(reward, axis=-1)  # Remove the last dimension to get (10, 10)
        # Add a decay factor and wider clipping to allow more evolution
        decay_factor = 0.99  # Slower decay for more gradual changes
        updated_interest = decay_factor * previous_state.get('interest') + 0.1 * reward
        updated_interest = tf.clip_by_value(updated_interest, -10.0, 10.0)  # Wider clipping range
        return value.Value(interest=updated_interest)