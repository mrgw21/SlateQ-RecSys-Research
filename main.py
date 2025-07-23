import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import gin
from absl import app
from absl import flags
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory as trajectory_lib

from src.runtimes.ecomm_runtime import ECommRuntime
from src.stories import ecomm_story
from src.agents.slateq_agent import SlateQAgent

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Paths to the config files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin parameter bindings.')

def main(argv):
    del argv
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

    num_users = 10
    slate_size = 5

    # Build RecSim NG network and runtime environment
    network = ecomm_story(num_users=num_users, num_items=100, slate_size=slate_size)
    rt = ECommRuntime(network=network)

    # Use unflattened action spec (slate of item indices)
    slate_spec = network.invariants()['rec_state'].get('slate')
    action_spec = slate_spec

    # Initialize agent with structured action_spec
    agent = SlateQAgent(
        time_step_spec=rt.time_step_spec(),
        action_spec=action_spec,
        num_users=num_users,
        num_topics=10,
        slate_size=slate_size,
        num_items=100
    )

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=1,
        max_length=10000
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=32,
        num_steps=2,
        single_deterministic_pass=False
    ).prefetch(3)

    num_episodes = 100
    loss_info = None

    for episode in range(num_episodes):
        time_step = rt.reset()
        episode_reward = 0.0

        for step in range(100):
            action_step = agent.policy.action(time_step)

            action = action_step.action

            next_time_step = rt.step(action)

            experience = trajectory_lib.from_transition(time_step, action_step, next_time_step)
            experience = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), experience)
            replay_buffer.add_batch(experience)
            episode_reward += next_time_step.reward.numpy().sum()
            time_step = next_time_step

            # Train on a batch from the replay buffer
            if replay_buffer.num_frames().numpy() > 32:
                for batch in dataset.take(1):
                    experience = batch[0]
                    loss_info = agent.train(experience)

            if step % 1000 == 0 and loss_info is not None:
                print(f"[Episode {episode}] Step {step} | Loss: {loss_info.loss.numpy():.4f}")

        print(f"[Episode {episode}] Total Reward: {episode_reward:.2f}")

    trajectory_result = rt.trajectory()
    user_state = trajectory_result.get('user_state')
    if user_state:
        interest = user_state.get('interest')
        if isinstance(interest, tf.Tensor):
            for i, value in enumerate(interest.numpy()):
                print(f"User {i} final interest: {value}")
        else:
            print("Interest:", interest)
    else:
        print("No user_state in final trajectory")

if __name__ == '__main__':
    app.run(main)
