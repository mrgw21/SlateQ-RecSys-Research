import gin
import tensorflow as tf

from recsim_ng.core import network, value, variable
from recsim_ng.core.variable import value as value_def
from recsim_ng.entities.state_models.static import StaticStateModel

from ..entities.ecomm_user import ECommUser
from ..entities.ecomm_recommender import ECommRecommender


class TensorSpec(value.FieldSpec):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def invariant(self):
        return {
            "type": "tensor",
            "shape": self.shape,
            "dtype": self.dtype.name if hasattr(self.dtype, "name") else str(self.dtype)
        }


class ItemStateModel(StaticStateModel):
    def __init__(self, num_items, num_topics, name="item_state"):
        super().__init__()
        self.num_items = num_items
        self.num_topics = num_topics
        self._name = name

    def initial_state(self):
        return value.Value(
            state=tf.random.uniform((self.num_items, self.num_topics), minval=-1.0, maxval=1.0)
        )

    def specs(self):
        return value.ValueSpec(
            state=TensorSpec(
                shape=(self.num_items, self.num_topics),
                dtype=tf.float32
            )
        )

    @property
    def name(self):
        return self._name

    def next_state(self, previous_state, action):
        return previous_state


@gin.configurable
def ecomm_story(num_users=10, num_items=100, slate_size=5):
    user_model = ECommUser(num_topics=10, num_users=num_users)
    recommender_model = ECommRecommender(num_topics=10, num_users=num_users, slate_size=slate_size)
    item_model = ItemStateModel(num_items=num_items, num_topics=10)

    # Dummy action/response specs
    action_spec = value.ValueSpec(
        act=TensorSpec(shape=(num_users, slate_size), dtype=tf.int32)
    )
    response_spec = value.ValueSpec(
        resp=TensorSpec(shape=(num_users,), dtype=tf.float32)
    )

    action_var = variable.Variable(name="action", spec=action_spec)
    action_var.initial_value = value_def(
        fn=lambda: value.Value(act=tf.zeros((num_users, slate_size), dtype=tf.int32)),
        dependencies=()
    )
    action_var.value = value_def(
        fn=lambda prev: prev,
        dependencies=(action_var.previous,)
    )

    response_var = variable.Variable(name="response", spec=response_spec)
    response_var.initial_value = value_def(
        fn=lambda: value.Value(resp=tf.zeros((num_users,), dtype=tf.float32)),
        dependencies=()
    )
    response_var.value = value_def(
        fn=lambda prev: prev,
        dependencies=(response_var.previous,)
    )

    # User
    user_var = variable.Variable(name="user_state", spec=user_model.specs())
    user_var.initial_value = value_def(fn=user_model.initial_state, dependencies=())
    user_var.value = value_def(
        fn=lambda state, response: user_model.next_state(state, response),
        dependencies=(user_var.previous, response_var)
    )

    # Recommender
    rec_var = variable.Variable(name="rec_state", spec=recommender_model.specs())
    rec_var.initial_value = value_def(fn=recommender_model.initial_state, dependencies=())
    rec_var.value = value_def(
        fn=lambda state, action: recommender_model.next_state(state, action),
        dependencies=(rec_var.previous, action_var)
    )

    # Item
    item_var = variable.Variable(name="item_state", spec=item_model.specs())
    item_var.initial_value = value_def(fn=item_model.initial_state, dependencies=())
    item_var.value = value_def(
        fn=lambda state, action: item_model.next_state(state, action),
        dependencies=(item_var.previous, action_var)
    )

    return network.Network(variables=[action_var, response_var, user_var, rec_var, item_var])
