from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity

def ecomm_story(num_users, num_items, slate_size):
    users = entity.Entity(name='users', state_model=ECommUser())
    items = entity.Entity(name='items', state_model=ECommItem())
    recommender = entity.Entity(name='recommender', state_model=ECommRecommender())
    return value.Value(users=users.state, items=items.state, recommender=recommender.state)