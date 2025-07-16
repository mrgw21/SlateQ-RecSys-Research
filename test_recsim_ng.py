import gin
from absl import app, flags
from recsim_ng.core import value, variable
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
from recsim_ng.core.network import Network as TFNetwork
from recsim_ng.core.value import FieldSpec, ValueSpec

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Gin files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings.')

@gin.configurable
def simple_story():
    state = variable.Variable(name='state', spec=ValueSpec(state=FieldSpec()))
    state.initial_value = variable.value(lambda: value.Value(state=tf.constant(0.0)))
    state.value = variable.value(lambda prev: value.Value(state=prev.get('state') + tf.constant(1.0)), (state.previous,))
    return [state]

def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)
    story = simple_story()
    network = TFNetwork(variables=story)
    rt = runtime.TFRuntime(network=network)
    trajectory = rt.trajectory(length=5)
    for step in trajectory:
        print(step)

if __name__ == '__main__':
    app.run(main)