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
        return value.ValueSpec(interest=value.FieldSpec())

    def initial_state(self):
        interest = tfd.Normal(loc=0., scale=1.).sample(sample_shape=(self.num_users, self.num_topics))
        print(f"Initial interest raw shape: {interest.shape}, values: {interest}")
        return value.Value(interest=interest)

    def response(self, user_state, slate, item_state):
        slate_indices = slate.get('slate')  # Shape: (10, slate_size) = (10, 5)
        print(f"slate_indices shape: {slate_indices.shape}, values: {slate_indices}")
        features = item_state.get('features')  # Shape: (num_topics,) = (10,)
        print(f"Features shape: {features.shape}, values: {features}")
        # Gather features using slate indices
        gathered_features = tf.gather(features, slate_indices)  # Should be (10, 5)
        print(f"Initial gathered features shape: {gathered_features.shape}, values: {gathered_features}")
        # Ensure gathered_features has the correct shape (10, 5, 1)
        gathered_features = tf.expand_dims(gathered_features, axis=-1)  # Shape: (10, 5, 1)
        print(f"Raw gathered features shape after expand: {gathered_features.shape}, values: {gathered_features}")
        # Squeeze any extra singleton dimensions
        gathered_features = tf.squeeze(gathered_features, axis=None)  # Remove extra singleton dims
        print(f"Gathered features shape after squeeze: {gathered_features.shape}, values: {gathered_features}")
        # Force reshape to (10, 5, 1) if the shape is incorrect
        if gathered_features.shape[-1] != 1:
            gathered_features = tf.reshape(gathered_features, [self.num_users, -1, 1])  # Reshape to (10, 5, 1)
        print(f"Gathered features final shape: {gathered_features.shape}, values: {gathered_features}")
        # Handle potential extra dimensions from runtime for interest
        interest = user_state.get('interest')  # Shape: (10, 10) or higher rank
        print(f"Raw interest shape: {interest.shape}, values: {interest}")
        interest = tf.squeeze(interest)  # Remove all singleton dimensions
        print(f"Interest shape after squeeze: {interest.shape}, values: {interest}")
        if len(interest.shape) == 3 and interest.shape[0] == self.num_users:
            interest = interest[:, :, 0]  # Take the first slice if it's a batch of interests
        elif len(interest.shape) == 3 and interest.shape[1] == self.num_topics:
            interest = interest[0, :, :]  # Take the first user if stacked by users
        elif len(interest.shape) != 2:
            raise ValueError(f"Interest shape {interest.shape} is not reducible to rank 2 after squeeze")
        print(f"Interest shape after reduction: {interest.shape}, values: {interest}")
        interest_expanded = tf.expand_dims(interest, axis=-1)  # Shape: (10, 10, 1)
        print(f"Interest expanded shape: {interest_expanded.shape}, values: {interest_expanded}")
        # Compute affinities using einsum with correct rank
        affinities = tf.einsum('bij,bkj->bi', interest_expanded, gathered_features)  # Shape: (10, 10, 1) @ (10, 5, 1) -> (10, 5)
        print(f"affinities shape: {affinities.shape}, values: {affinities}")
        choice = tfd.Categorical(logits=affinities).sample()  # Shape: (10,)
        print(f"choice shape: {choice.shape}, values: {choice}")
        # Fix reward shape to (10, 1)
        reward = tf.expand_dims(tf.gather(affinities, choice, batch_dims=0, axis=1), axis=-1)  # Shape: (10, 1)
        print(f"reward shape: {reward.shape}, values: {reward}")
        return value.Value(choice=choice, reward=reward)

    def next_state(self, previous_state, response):
        reward = tf.clip_by_value(response.get('reward'), -1.0, 1.0)  # Clip reward to prevent extreme values
        print(f"Next state reward shape: {reward.shape}, values: {reward}")  # Debug reward
        # Add a decay factor and tighter clipping to stabilize interest
        decay_factor = 0.9  # Decay previous interest to prevent accumulation
        updated_interest = decay_factor * previous_state.get('interest') + 0.1 * reward
        updated_interest = tf.clip_by_value(updated_interest, -5.0, 5.0)  # Tighter clipping range
        print(f"Updated interest shape: {updated_interest.shape}, values: {updated_interest}")  # Debug updated interest
        return value.Value(interest=updated_interest)