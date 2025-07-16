import gin
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from recsim_ng.core import value, variable
from recsim_ng.lib.tensorflow import entity

from src.entities.ecomm_user import ECommUser
from src.entities.ecomm_item import ECommItem
from src.entities.ecomm_recommender import ECommRecommender

@gin.configurable
def ecomm_story(num_users, num_items, slate_size):
    user_model = ECommUser(num_topics=10, num_users=num_users)
    item_model = ECommItem(num_topics=10)
    recommender_model = ECommRecommender(num_topics=10)

    user_state = variable.Variable(name='user_state', spec=user_model.specs())
    item_state = variable.Variable(name='item_state', spec=item_model.specs())
    rec_state = variable.Variable(name='rec_state', spec=recommender_model.specs())
    slate = variable.Variable(name='slate', spec=value.ValueSpec(slate=value.FieldSpec()))
    response = variable.Variable(name='response', spec=value.ValueSpec(choice=value.FieldSpec(), reward=value.FieldSpec()))

    user_state.initial_value = variable.value(user_model.initial_state)
    item_state.initial_value = variable.value(item_model.initial_state)
    rec_state.initial_value = variable.value(recommender_model.initial_state)
    slate.initial_value = variable.value(lambda: value.Value(slate=tf.zeros((slate_size,), dtype=tf.int32)))
    response.initial_value = variable.value(lambda: value.Value(choice=tf.constant(0, dtype=tf.int32), reward=tf.constant(0.0)))

    slate.value = variable.value(
        lambda rec_prev, user_prev: recommender_model.select_slate(rec_prev, user_prev, slate_size),
        (rec_state.previous, user_state.previous)
    )
    response.value = variable.value(user_model.response, (user_state.previous, slate, item_state.previous))
    user_state.value = variable.value(user_model.next_state, (user_state.previous, response))
    item_state.value = variable.value(item_model.next_state, (item_state.previous, response))
    rec_state.value = variable.value(recommender_model.next_state, (rec_state.previous, response))

    # Debug: Print initial shapes
    initial_user_state = user_model.initial_state()
    initial_item_state = item_model.initial_state()
    initial_rec_state = recommender_model.initial_state()
    print(f"Initial user_state interest shape: {initial_user_state.get('interest').shape}")
    print(f"Initial item_state features shape: {initial_item_state.get('features').shape}")
    print(f"Initial rec_state rec_features shape: {initial_rec_state.get('rec_features').shape}")
    print(f"Initial slate shape: {(slate_size,)}")
    print(f"Initial response choice shape: ()")
    print(f"Initial response reward shape: ()")

    return [user_state, item_state, rec_state, slate, response]