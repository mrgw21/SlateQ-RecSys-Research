import gin
import tensorflow as tf
from absl import app, flags
from src.stories.ecomm_story import ecomm_story
from src.runtimes.ecomm_runtime import ECommRuntime
from recsim_ng.core.network import Network as TFNetwork

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Gin files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings.')

def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)
    story = ecomm_story(num_users=10, num_items=100, slate_size=5)
    network = TFNetwork(variables=story)
    rt = ECommRuntime(network=network)
    trajectory = rt.trajectory(num_steps=5000)  # Increased to 5000 to test convergence
    user_state = trajectory.get('user_state')
    if user_state:
        interest = user_state.get('interest')
        if isinstance(interest, tf.Tensor):
            for i, value in enumerate(interest.numpy()):
                print(f"User {i} final interest: {value}")
        else:
            print(f"Interest: {interest}")
    else:
        print("No user_state in trajectory")

if __name__ == '__main__':
    app.run(main)