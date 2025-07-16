import gin
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

    # Define Variables with specs from models
    user_state = variable.Variable(name='user_state', spec=user_model.specs())
    item_state = variable.Variable(name='item_state', spec=item_model.specs())
    rec_state = variable.Variable(name='rec_state', spec=recommender_model.specs())
    slate = variable.Variable(name='slate', spec=variable.ValueSpec(space=variable.DiscreteSpace(slate_size)))
    response = variable.Variable(name='response', spec=variable.ValueSpec(space=variable.DiscreteSpace(1)))

    # Initial bindings
    user_state.initial_value = variable.value(user_model.initial_state)
    item_state.initial_value = variable.value(item_model.initial_state)
    rec_state.initial_value = variable.value(recommender_model.initial_state)

    # Dynamics (transitions)
    slate.value = variable.value(recommender_model.select_slate, (rec_state.previous, user_state.previous))
    response.value = variable.value(user_model.response, (user_state.previous, slate.value, item_state.previous))
    user_state.value = variable.value(user_model.next_state, (user_state.previous, response.value))
    item_state.value = variable.value(item_model.next_state, (item_state.previous, response.value))
    rec_state.value = variable.value(recommender_model.next_state, (rec_state.previous, response.value))

    # Return list of Variables for Network
    return [user_state, item_state, rec_state, slate, response]