import gin
import tensorflow as tf

from recsim_ng.core import network, value, variable
from recsim_ng.core.variable import value as value_def
from recsim_ng.entities.state_models.static import StaticStateModel

from ..entities.ecomm_user import ECommUser
from ..entities.ecomm_recommender import ECommRecommender


class TensorFieldSpec(value.FieldSpec):
    def __init__(self, shape, dtype=tf.float32):
        super().__init__()
        self._shape = shape
        self._dtype = dtype

    def invariant(self):
        return tf.TensorSpec(shape=self._shape, dtype=self._dtype)

class ItemStateModel(StaticStateModel):
    def __init__(self, num_items, num_topics, name="item_state"):
        super().__init__()
        self.num_items = num_items
        self.num_topics = num_topics
        self._name = name

    def specs(self):
        return value.ValueSpec(
            features=TensorFieldSpec(
                shape=(self.num_items, self.num_topics),
                dtype=tf.float32
            )
        )

    def initial_state(self):
        return value.Value(
            features=tf.random.uniform(
                (self.num_items, self.num_topics),
                minval=-1.0,
                maxval=1.0
            )
        )

    def next_state(self, previous_state, action):
        return previous_state

    @property
    def name(self):
        return self._name


@gin.configurable
def ecomm_story(num_users=10, num_items=100, slate_size=5):
    user_model = ECommUser(num_topics=10, num_users=num_users)
    rec_model = ECommRecommender(num_topics=10, num_users=num_users, slate_size=slate_size)
    item_model = ItemStateModel(num_items=num_items, num_topics=10)

    action_spec = value.ValueSpec(
        act=TensorFieldSpec(shape=(num_users, slate_size), dtype=tf.int32)
    )
    action_var = variable.Variable(name="action", spec=action_spec)
    action_var.initial_value = value_def(fn=lambda: value.Value(
        act=tf.zeros((num_users, slate_size), dtype=tf.int32)), dependencies=())
    action_var.value = value_def(fn=lambda prev: prev, dependencies=(action_var.previous,))

    item_var = variable.Variable(name="item_state", spec=item_model.specs())
    item_var.initial_value = value_def(fn=item_model.initial_state, dependencies=())
    item_var.value = value_def(fn=lambda prev: prev, dependencies=(item_var.previous,))

    user_var = variable.Variable(name="user_state", spec=user_model.specs())
    user_var.initial_value = value_def(fn=user_model.initial_state, dependencies=())

    response_spec = value.ValueSpec(
        reward=TensorFieldSpec(shape=(num_users,), dtype=tf.float32),
        choice=TensorFieldSpec(shape=(num_users,), dtype=tf.int32)
    )
    response_var = variable.Variable(name="response", spec=response_spec)
    response_var.initial_value = value_def(fn=lambda: value.Value(
        reward=tf.zeros((num_users,), dtype=tf.float32),
        choice=tf.zeros((num_users,), dtype=tf.int32)
    ), dependencies=())

    response_var.value = value_def(
        fn=lambda user_state, action, item_state:
            user_model.response(
                user_state,
                value.Value(slate=action.get("act")),
                item_state
            ),
        dependencies=(user_var, action_var, item_var.previous)
    )

    user_var.value = value_def(
        fn=lambda state, response: user_model.next_state(state, response),
        dependencies=(user_var.previous, response_var.previous)
    )

    rec_var = variable.Variable(name="rec_state", spec=rec_model.specs())
    rec_var.initial_value = value_def(fn=rec_model.initial_state, dependencies=())
    rec_var.value = value_def(
        fn=lambda prev_state, action: rec_model.next_state(prev_state, action),
        dependencies=(rec_var.previous, action_var)
    )

    return network.Network(variables=[
        action_var,
        item_var,
        user_var,
        response_var,
        rec_var
    ])

