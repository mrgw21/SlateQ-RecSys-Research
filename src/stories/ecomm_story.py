import gin
from recsim_ng.core import network
import tensorflow as tf
from ..entities.ecomm_user import ECommUser
from ..entities.ecomm_recommender import ECommRecommender
from recsim_ng.entities.state_models.static import StaticStateModel
from recsim_ng.core import value

class ItemStateModel(StaticStateModel):
    def __init__(self, num_items, num_topics):
        super().__init__()
        self.num_items = num_items
        self.num_topics = num_topics

    def initial_state(self):
        return value.Value(state=tf.random.uniform((self.num_items, self.num_topics), minval=-1.0, maxval=1.0))

    def specs(self):
        return value.ValueSpec(state=value.FieldSpec(shape=(self.num_items, self.num_topics), dtype=tf.float32))

@gin.configurable
def ecomm_story(num_users=10, num_items=100, slate_size=5):
    # Define user state model
    user = ECommUser(num_topics=10, num_users=num_users)

    # Define recommender state model
    recommender = ECommRecommender(num_topics=10, num_users=num_users, slate_size=slate_size)

    # Define item state model
    item_state = ItemStateModel(num_items=num_items, num_topics=10)

    # Compose the network
    return network.Network(
        variables={
            "user_state": user,
            "rec_state": recommender,
            "item_state": item_state
        }
    )