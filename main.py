import gin
from absl import app, flags
from src.stories import ecomm_story
from src.runtimes import ecomm_runtime

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Gin files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings.')

def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)
    story = ecomm_story(num_users=10, num_items=100, slate_size=5)
    rt = ecomm_runtime.ECommRuntime(story, num_steps=1000)
    trajectory = rt.trajectory()
    print(trajectory)

if __name__ == '__main__':
    app.run(main)