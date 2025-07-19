from recsim_ng.core import network
from recsim_ng.entities.state_models import DynamicStateModel  # Use specific class
from recsim_ng.core import ActionModel  # Updated import
import tensorflow as tf
from ..entities.ecomm_user import ECommUser
from ..entities.ecomm_recommender import ECommRecommender

def ecomm_story(num_users=10, num_items=100, slate_size=5):
    # Define user state model
    user = DynamicStateModel.create(
        ECommUser,
        num_topics=10,
        num_users=num_users
    )

    # Define recommender action model
    recommender = ActionModel.create(
        ECommRecommender,
        num_topics=10,
        num_users=num_users,
        slate_size=slate_size
    )

    # Define item state model
    item_state = DynamicStateModel.create(
        lambda: tf.random.uniform((num_items, 10), minval=-1.0, maxval=1.0),
        name="item_state"
    )

    # Compose the network
    return network.Network(
        variables={
            "user_state": user,
            "rec_state": recommender,
            "item_state": item_state
        }
    )