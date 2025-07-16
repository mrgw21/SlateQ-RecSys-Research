import gin
from absl import app, flags
from src.stories.ecomm_story import ecomm_story
from src.runtimes.ecomm_runtime import ECommRuntime

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Gin files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings.')

def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)
    story = ecomm_story(num_users=10, num_items=100, slate_size=5)
    rt = ECommRuntime(story)
    trajectory = rt.trajectory()  # Uses num_steps from class
    print(trajectory)

if __name__ == '__main__':
    app.run(main)